# Lab AI Engineer sur mini-PC Ryzen

Ce lab fait tourner un LLM sur ton mini-PC et te permet de l’utiliser depuis ton portable ou ton téléphone. Le projet de formation est un assistant qui répond en français à partir de runbooks et cite les passages utilisés. La procédure part d’un Ryzen 5000, de 32 Go de RAM et d’un disque de 500 Go. Le processeur exact, le type de disque et la présence d’un GPU dédié restent à vérifier.

Le [parcours AI Engineer de trois jours](FORMATION.md) s’appuie sur cette installation.

## Architecture et dimensionnement

~~~mermaid
flowchart LR
    Laptop["Portable ou téléphone"] -->|"Tailscale · HTTPS"| UI["Open WebUI"]
    Laptop -->|"Tailscale · tunnel SSH"| API["API FastAPI"]
    UI --> LLM["Ollama · Qwen3 + BGE-M3"]
    API --> LLM
    API --> DB["Qdrant · index de runbooks"]
    Docs["PDF texte, Markdown et TXT"] -->|"Indexation locale"| DB
~~~

| Élément | Réglage de départ |
| --- | --- |
| Système | Ubuntu Server 24.04 LTS en VM si le mini-PC est sous Proxmox. Ubuntu installé directement fonctionne aussi. |
| VM `lab-ai` | 6 vCPU si l’hôte le permet, 16 Go de RAM fixes, disque VirtIO SCSI de 100 à 120 Gio, un vNIC VirtIO sur `vmbr0`. |
| Inférence | CPU, contexte de 4 096 tokens, une génération simultanée par modèle, jusqu’à deux modèles chargés selon la mémoire disponible. |
| Modèles | `qwen3:4b` pour répondre et `bge-m3:latest` pour indexer. `qwen3:8b` est le modèle de comparaison après les mesures. |
| Accès | Open WebUI en HTTPS sur ton réseau privé Tailscale. SSH par clé et tunnels pour les API. |

`qwen3:4b` et `qwen3:8b` sont sous licence Apache 2.0. Ce sont des modèles préentraînés dont tu héberges les poids et l’inférence. Leurs téléchargements Q4_K_M font environ **2,5 Go** et **5,2 Go**. `bge-m3`, modèle d’embeddings multilingue sous licence MIT, ajoute **1,2 Go**. La RAM utilisée comprend aussi le contexte et les services. La vitesse et l’intérêt du 8B se mesurent sur ta machine. [Qwen3 4B](https://ollama.com/library/qwen3:4b), [Qwen3 8B](https://ollama.com/library/qwen3:8b), [BGE-M3 dans Ollama](https://ollama.com/library/bge-m3), [fiche et licence BGE-M3](https://huggingface.co/BAAI/bge-m3).

Les conversations, les documents, les embeddings et la génération passent par les composants du mini-PC. L’installation et les mises à jour téléchargent des images et des modèles. Tailscale fournit la coordination du réseau privé. [Mode local Ollama](https://docs.ollama.com/faq), [Tailscale Serve](https://tailscale.com/docs/features/tailscale-serve).

## 1. Préparer le mini-PC

Sur le **mini-PC physique**, vérifie les ressources et les VM déjà présentes.

~~~bash
lscpu
free -h
lsblk -d -o NAME,SIZE,ROTA,MODEL
~~~

`ROTA=0` désigne normalement un SSD ou NVMe, `ROTA=1` un disque mécanique. Avec un HDD, prévois surtout des temps de chargement et d’indexation plus longs. Branche le mini-PC en Ethernet et désactive sa mise en veille pour qu’il reste joignable.

- **Proxmox déjà en place.** Dans **Summary**, relève la mémoire et le stockage disponibles. Crée la VM `lab-ai` avec l’ISO Ubuntu Server 24.04 amd64. Dans **CPU**, sélectionne le type `host`. Désactive le ballooning pour ce lab, active **QEMU Guest Agent**, puis **Start at boot**. Utilise le bridge LAN existant, habituellement `vmbr0`. Laisse idéalement au moins 30 à 50 Gio réellement libres sur le stockage de l’hôte après allocation. Si d’autres VM tournent, commence à 4 vCPU et 12 Go avec le modèle 4B, puis augmente selon la marge constatée. [Gestion des VM Proxmox](https://github.com/proxmox/pve-docs/blob/master/qm.adoc).
- **Ubuntu 24.04 déjà en place ou installation directe souhaitée.** Passe à la préparation Ubuntu. Sur un disque vierge, tu peux aussi installer Proxmox puis créer la VM selon le point précédent. Pour une installation neuve, sauvegarde d’abord tes données et vérifie le disque cible dans l’installateur. [ISO Ubuntu Server 24.04](https://releases.ubuntu.com/24.04/), [ISO Proxmox VE](https://www.proxmox.com/en/downloads/proxmox-virtual-environment/iso).

### Préparer Ubuntu et la clé SSH

Dans l’installateur, donne à Ubuntu le nom `lab-ai`, crée le compte `labadmin` et sélectionne **Install OpenSSH server**. Réserve le bail DHCP de cette machine sur la box si tu veux retrouver facilement son IP locale. `192.168.1.60` dans les exemples est une adresse à remplacer par celle affichée par `ip -br -4 addr`.

**Dans Ubuntu** (VM ou installation directe).

~~~bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y ca-certificates curl git jq openssh-server openssl unzip python3-venv
ip -br -4 addr
free -h
df -h /
~~~

En cas de partition racine LVM, vérifie que `df -h /` te donne l’espace voulu pour Docker. Dans une VM neuve, une racine de l’ordre de 80 Gio ou davantage laisse de la marge pour les images, modèles et index. Le programme d’installation peut n’allouer qu’une partie du groupe de volumes.

Dans la VM Proxmox, installe aussi `sudo apt install -y qemu-guest-agent`. Si l’agent reste indisponible, arrête et redémarre la VM après avoir activé l’option **QEMU Guest Agent** sur Proxmox.

**Sur le portable**, utilise une clé existante ou crée une paire avec `ssh-keygen -t ed25519`. Dans la console Ubuntu, connecté en `labadmin`, ajoute la ligne de ta clé publique `.pub` à `~/.ssh/authorized_keys`, puis teste la connexion depuis le portable.

~~~bash
# Dans Ubuntu.
install -d -m 700 ~/.ssh
touch ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
nano ~/.ssh/authorized_keys

# Sur le portable.
ssh -o PreferredAuthentications=publickey labadmin@192.168.1.60
~~~

Une fois la connexion par clé confirmée, garde la première session ouverte. Dans Ubuntu, applique ces réglages, puis reconnecte-toi dans un second terminal pour vérifier.

~~~bash
sudo tee /etc/ssh/sshd_config.d/00-lab-ai.conf >/dev/null <<'EOF'
PermitRootLogin no
PasswordAuthentication no
KbdInteractiveAuthentication no
EOF
sudo sshd -t && sudo systemctl reload ssh
~~~

## 2. Installer Docker et lancer le chat local

Sur **Ubuntu 24.04**, si Docker Engine et le plugin Compose sont déjà installés, vérifie leurs versions et saute l’ajout du dépôt. Utilise Docker Engine 28 ou plus récent pour le comportement documenté des ports liés à `127.0.0.1`. Sur une installation neuve, installe les paquets officiels suivants. [Installation Docker sur Ubuntu](https://docs.docker.com/engine/install/ubuntu/), [Publication des ports Docker](https://docs.docker.com/engine/network/port-publishing/).

~~~bash
sudo install -d -m 0755 /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<'EOF'
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: noble
Components: stable
Architectures: amd64
Signed-By: /etc/apt/keyrings/docker.asc
EOF
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo docker version
sudo docker compose version
~~~

### Récupérer les fichiers depuis GitHub

Depuis ton portable, ouvre le dépôt avec ton compte GitHub et choisis **Code → Download ZIP**. Dans le dossier où le navigateur a enregistré l’archive, envoie-la sur Ubuntu.

~~~bash
scp lab-ai-engineer-main.zip labadmin@192.168.1.60:~/
~~~

Pour une première installation, extrais l’archive dans Ubuntu puis donne au dossier le nom `lab-ai`.

~~~bash
cd ~
unzip lab-ai-engineer-main.zip
mv -T lab-ai-engineer-main lab-ai
~~~

Si Ubuntu dispose déjà d’une clé SSH autorisée sur ton compte GitHub et que le dossier `~/lab-ai` est libre, tu peux aussi cloner le dépôt directement.

~~~bash
git clone git@github.com:ahouab/lab-ai-engineer.git ~/lab-ai
~~~

Si tu as déjà extrait le kit dans `~/lab-ai`, utilise directement ce dossier. Lance les commandes Compose de la procédure depuis ce répertoire. Le secret généré dans `.env` permet à l’interface de conserver ses sessions lors des redémarrages.

~~~bash
chmod 700 ~/lab-ai
cd ~/lab-ai
(umask 077; test -e .env || printf 'WEBUI_SECRET_KEY=%s\n' "$(openssl rand -hex 32)" > .env)
sudo docker compose config --quiet
sudo docker compose up -d ollama webui
sudo docker compose exec ollama ollama pull qwen3:4b
sudo docker compose exec ollama ollama list
curl -fsS http://127.0.0.1:11434/api/tags | jq
curl -fsS http://127.0.0.1:3000/health
~~~

Si le dernier appel échoue pendant le démarrage, attends que `sudo docker compose logs --tail=40 webui` montre un serveur prêt, puis relance le test. Les conteneurs utilisent des versions fixées dans [compose.yaml](compose.yaml). [Ollama sur Docker](https://docs.ollama.com/docker), [démarrage Open WebUI](https://docs.openwebui.com/getting-started/quick-start/).

Depuis le **portable**, ouvre un tunnel le temps de créer l’administrateur Open WebUI.

~~~bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:3000:127.0.0.1:3000 labadmin@192.168.1.60
~~~

Ouvre [http://127.0.0.1:3000](http://127.0.0.1:3000), crée ton compte administrateur et vérifie dans **Admin Panel → Settings → Authentication** que les nouvelles inscriptions sont désactivées. Dans un chat, sélectionne `qwen3:4b` et demande « Explique en trois points le rôle d’une sonde readiness Kubernetes ». Pendant ou après la réponse, `sudo docker compose exec ollama ollama ps` doit indiquer un modèle chargé sur CPU. Ferme le tunnel de démarrage avec `Ctrl+C`. [Connexion d’Open WebUI à Ollama](https://docs.openwebui.com/getting-started/quick-start/connect-a-provider/starting-with-ollama/).

### Créer ton assistant local

Le [Modelfile](Modelfile) fourni configure un assistant francophone à partir de Qwen3. Tu peux versionner son prompt système et ses paramètres dans Git.

~~~bash
sudo docker compose cp Modelfile ollama:/tmp/Modelfile
sudo docker compose exec ollama ollama create assistant-runbooks -f /tmp/Modelfile
sudo docker compose exec ollama ollama run assistant-runbooks --think=false "Explique la différence entre RTO et RPO."
~~~

Recharge la page Open WebUI pour sélectionner `assistant-runbooks`. Cette personnalisation utilise les mêmes poids de base. [Référence Modelfile](https://docs.ollama.com/modelfile).

Si tu disposes de ton propre modèle au format GGUF compatible avec Ollama, tu peux l’importer. Copie le fichier dans le conteneur avec `sudo docker compose cp modele.gguf ollama:/tmp/modele.gguf`, puis utilise `FROM /tmp/modele.gguf` dans une copie du Modelfile et crée un modèle sous un nouveau nom. [Import GGUF](https://docs.ollama.com/modelfile#build-from-a-gguf-model).

## 3. Accéder à distance

### Installer le réseau privé

Dans **Ubuntu 24.04**, installe Tailscale et rattache le mini-PC à ton réseau privé, appelé *tailnet*. Ouvre l’URL de connexion affichée et utilise ton compte. Installe le client sur ton portable ou ton téléphone et connecte-le au même réseau. [Dépôt Tailscale pour Ubuntu 24.04](https://pkgs.tailscale.com/stable/#ubuntu-noble), [clients Tailscale](https://tailscale.com/download).

~~~bash
sudo install -d -m 0755 /usr/share/keyrings
sudo curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/noble.noarmor.gpg -o /usr/share/keyrings/tailscale-archive-keyring.gpg
sudo curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/noble.tailscale-keyring.list -o /etc/apt/sources.list.d/tailscale.list
sudo apt update
sudo apt install -y tailscale
sudo systemctl enable --now tailscaled
sudo tailscale up --hostname=lab-ai
tailscale status
tailscale ip -4
~~~

**Pour le chat**, après création du compte Open WebUI, publie le port local avec Tailscale Serve. Si Tailscale te le demande, active **MagicDNS** et **HTTPS Certificates** dans la console DNS. Le certificat publie le nom DNS du mini-PC dans les journaux de transparence TLS. Tu peux rester sur le tunnel SSH si tu veux éviter cette publication. [Commande Serve](https://tailscale.com/docs/reference/tailscale-cli/serve), [HTTPS et nom DNS](https://tailscale.com/docs/how-to/set-up-https-certificates).

~~~bash
sudo tailscale serve --bg http://127.0.0.1:3000
tailscale serve status
~~~

Sur le **portable ou le téléphone**, ouvre l’URL HTTPS exacte renvoyée par la commande, par exemple `https://lab-ai.exemple.ts.net`. Sur ton téléphone en 4G/5G, vérifie l’accès avec Tailscale actif, puis refais un essai avec Tailscale coupé. L’URL doit fonctionner uniquement avec un appareil autorisé. Sur une VM neuve, il suffit de cette configuration côté box, sans redirection de port ni Tailscale Funnel.

**Pour SSH**, depuis le portable, utilise le nom MagicDNS ou l’IP renvoyée par `tailscale ip -4`. Sur un tailnet partagé, applique des règles d’accès qui réservent SSH et HTTPS à tes comptes ou appareils autorisés. [MagicDNS](https://tailscale.com/docs/features/magicdns), [règles d’accès Tailscale](https://tailscale.com/docs/features/access-control/grants).

~~~bash
ssh labadmin@lab-ai
~~~

Sur une VM dédiée avec UFW, tu peux limiter le SSH aux appareils du LAN et à Tailscale. Depuis la console de la VM, adapte le sous-réseau d’exemple, applique les règles et reteste les deux connexions. Si un pare-feu est déjà en place, intègre ces règles à sa configuration. [Pare-feu Ubuntu](https://documentation.ubuntu.com/server/how-to/security/firewalls/).

~~~bash
sudo apt install -y ufw
sudo ufw allow from 192.168.1.0/24 to any port 22 proto tcp
sudo ufw allow in on tailscale0 to any port 22 proto tcp
sudo ufw enable
sudo ufw status verbose
~~~

### Accéder aux API depuis le portable

Une fois le RAG démarré à l’étape suivante, garde ce tunnel ouvert. Il te permet d’appeler les deux API depuis un IDE ou un terminal. Le tunnel est aussi utilisable avec l’adresse LAN à domicile. Pour accéder au chat par SSH, ajoute `-L 127.0.0.1:3000:127.0.0.1:3000` et ouvre `http://127.0.0.1:3000`.

~~~bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8001:127.0.0.1:8001 -L 127.0.0.1:11434:127.0.0.1:11434 labadmin@lab-ai
~~~

| Depuis le portable | Adresse avec le tunnel ouvert |
| --- | --- |
| Documentation interactive RAG | [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs) |
| API RAG | `http://127.0.0.1:8001/ask` et `http://127.0.0.1:8001/search` |
| API Ollama native | `http://127.0.0.1:11434/api/chat` |

L’API RAG et l’API Ollama reposent sur l’authentification SSH pour cet usage personnel. Compose publie les ports sur la seule boucle locale. Qdrant reste sur le réseau Docker privé du RAG. L’interface interactive FastAPI charge ses ressources d’affichage depuis un CDN standard. Pour un poste hors ligne, utilise directement `curl`. [Documentation FastAPI](https://fastapi.tiangolo.com/how-to/custom-docs-ui-assets/).

## 4. Indexer les runbooks et poser une question

Dans Ubuntu, télécharge le modèle d’embeddings utilisé par les deux parcours.

~~~bash
sudo docker compose exec ollama ollama pull bge-m3
~~~

**Option depuis le navigateur.** Dans Open WebUI, attache par exemple `corpus/sauvegarde.md` à un chat et demande l’horaire de sauvegarde. L’interface utilise son propre index de documents avec les embeddings Ollama configurés par Compose. [RAG documentaire dans Open WebUI](https://docs.openwebui.com/features/chat-conversations/rag/).

**Parcours AI Engineer.** Lance le profil RAG dans Ubuntu pour manipuler toi-même les extraits, les vecteurs, la recherche et l’API. Le corpus fourni contient des runbooks fictifs en français.

~~~bash
cd ~/lab-ai
sudo docker compose --profile rag up -d --build
curl -fsS http://127.0.0.1:8001/healthz
sudo docker compose --profile rag exec -T rag python rag.py
curl -fsS http://127.0.0.1:8001/readyz
~~~

Si `/healthz` ou Qdrant est encore en démarrage, vérifie `sudo docker compose --profile rag ps` puis réessaie. L’indexation crée une nouvelle collection et la rend active après traitement complet. Le dossier `corpus/` est monté en lecture seule dans l’API.

Tu peux exécuter la suite dans Ubuntu ou sur le portable avec le tunnel ouvert. Sous Windows, utilise `curl.exe` ou le formulaire **Try it out** de [la documentation interactive](http://127.0.0.1:8001/docs).

~~~bash
curl -fsS http://127.0.0.1:8001/search -H 'Content-Type: application/json' -d '{"question":"Quel jour et à quelle heure se fait la sauvegarde complète ?"}' | jq
curl -fsS --max-time 510 http://127.0.0.1:8001/ask -H 'Content-Type: application/json' -d '{"question":"Quel jour et à quelle heure se fait la sauvegarde complète ?"}' | jq '{verdict, answer, sources, retrieval_ms, generation_ms}'
curl -fsS --max-time 510 http://127.0.0.1:8001/ask -H 'Content-Type: application/json' -d '{"question":"Dans quelle région AWS se trouve le cluster de production ?"}' | jq '{verdict, answer, sources}'
~~~

Le premier appel affiche les trois passages les plus proches. Pour la question de sauvegarde, la réponse attendue est **samedi à 03 h 40**, avec `sauvegarde.md` en source. La question AWS doit conduire à une abstention. Le code vérifie que chaque citation renvoie à un passage réellement récupéré. Lis le passage pour vérifier qu’il justifie la réponse. [Embeddings Ollama](https://docs.ollama.com/api/embed), [recherche Qdrant](https://api.qdrant.tech/api-reference/search/query-points).

Pour ajouter tes documents, copie tes fichiers Markdown, TXT UTF-8 ou PDF avec couche texte dans `corpus/`. Un PDF scanné nécessite d’abord un OCR local. Vérifie que l’utilisateur du conteneur peut lire les fichiers choisis. Si nécessaire, utilise par exemple `chmod a+r corpus/mon-runbook.pdf`, en gardant `~/lab-ai` accessible uniquement à ton compte Ubuntu. Relance `python rag.py` selon la commande ci-dessus après tout ajout, retrait ou changement du découpage. Les anciennes collections restent en base pour un retour arrière. Évite de les accumuler lors des essais sur un corpus volumineux.

## 5. Vérifier, sauvegarder et dépanner

**Recette.** Vérifie une conversation sur ton téléphone en 4G/5G, une requête RAG avec passage cité, une question sans réponse et la reconnexion après un redémarrage planifié. Depuis le LAN, les ports `3000`, `8001` et `11434` de l’IP Ubuntu doivent être inaccessibles directement. Sur Ubuntu, `sudo docker compose --profile rag ps` doit montrer les liaisons `127.0.0.1`.

**Évaluation.** Depuis Ubuntu, exécute le jeu de questions synthétiques fourni. Le rapport détaille les réponses, la présence de la source dans les trois résultats, la latence médiane et le P95 indicatif. Le code de sortie est 1 si un cas échoue. [Les exercices et critères sont dans le parcours de formation](FORMATION.md).

~~~bash
sudo docker compose --profile rag exec -T rag python eval_lab.py > resultats-4b.json
jq '.summary' resultats-4b.json
~~~

**Sauvegarde.** Sous Proxmox, programme une sauvegarde de la VM vers un stockage séparé du disque de 500 Go, par exemple un NAS ou Proxmox Backup Server. Sauvegarde les volumes Ollama, Open WebUI et Qdrant, les documents et le secret `.env` via la VM. Pour Ubuntu installé directement, arrête les conteneurs avec `sudo docker compose --profile rag stop` et sauvegarde le dossier et les volumes nommés sur un autre support. Sur Ubuntu standard, le chemin des volumes s’obtient avec `sudo docker volume inspect lab-ai_webui_data` et ses équivalents `lab-ai_ollama_data` et `lab-ai_qdrant_data`. Relance les services avec `sudo docker compose --profile rag up -d`. Essaie une restauration isolée avant de reconnecter une VM clonée à ton réseau privé. [Sauvegardes Proxmox](https://github.com/proxmox/pve-docs/blob/master/vzdump.adoc), [sauvegarder des volumes Docker](https://docs.docker.com/engine/storage/volumes/#back-up-restore-or-migrate-data-volumes).

| Symptôme | Vérifications |
| --- | --- |
| Chat inaccessible | Dans Ubuntu, appelle `curl -fsS http://127.0.0.1:3000/health`, puis consulte `tailscale status` et `tailscale serve status`. Sur le portable, vérifie que Tailscale est actif et que tu utilises le bon nom. |
| Aucun modèle dans le chat | Vérifie `sudo docker compose exec ollama ollama list` et le réglage **Connections → Ollama** dans l’interface. L’URL interne doit être `http://ollama:11434`. |
| RAG indisponible | Vérifie `/readyz`, télécharge `bge-m3` et `qwen3:4b`, puis relance l’indexation si l’alias `runbooks` est absent. Consulte `sudo docker compose --profile rag logs --tail=60 rag qdrant ollama`. |
| Inférence lente ou swap | Observe `free -h`, `sudo docker stats --no-stream` et `ollama ps` dans le conteneur. Compare un premier appel à un appel modèle déjà chargé. Garde le 4B et réduis le contexte ou la taille des réponses si nécessaire. |
| Disque qui se remplit | Contrôle `df -h /`, le stockage Proxmox et `sudo docker system df`. Les volumes mémorisent modèles, conversations et index. Pour un arrêt courant, utilise `stop` et `up -d` pour reprendre. |

Les réglages du kit visent un utilisateur sur CPU. Fais les mesures J1 et J2 sur la machine avant de retenir une cible de latence, de choisir le modèle 8B ou d’ajouter plusieurs utilisateurs.
