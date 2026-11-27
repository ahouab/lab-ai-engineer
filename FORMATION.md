# Parcours pratique AI Engineer · 3 jours, 21 heures

Le projet fil rouge est **un assistant SRE qui répond en français à partir de runbooks, cite ses sources et s’abstient quand les documents sont insuffisants**. En tant qu’AI Engineer, tu construis et exploites la chaîne complète. Tu choisis un modèle, conçois le flux de données, intègres les outils, mesures les erreurs et déploies le service. Les exercices réutilisent tes repères Cloud et SRE comme les sondes, la résilience, la gestion des droits et la mesure des performances.

Suis la [procédure d’installation](README.md). Prépare un portable avec SSH, un éditeur Python, Git et un accès au mini-PC. Les exemples utilisent Python 3.12, les API JSON d’Ollama et Qdrant et des documents fictifs. Lance les commandes sur Ubuntu, depuis `~/lab-ai`, sauf indication sur l’accès distant.

## J1 · Héberger un LLM et mesurer son comportement · 7 h

| Durée | Travail et résultat observé |
| --- | --- |
| 1 h 30 | Préparer la VM ou Ubuntu, lancer Docker et Ollama, télécharger Qwen3 4B. Un appel local fonctionne. |
| 1 h | Activer le VPN et Open WebUI. Depuis un autre réseau, le chat s’ouvre sur un appareil autorisé. |
| 1 h | Relier tokenisation, attention, Transformer, génération et embeddings au traitement d’une requête. Estimer l’effet d’une fenêtre de contexte plus longue sur la mémoire. |
| 1 h 30 | Tester température, prompt système et exemples. Créer `assistant-runbooks`, puis comparer une sortie libre et une sortie JSON contrainte. |
| 1 h | Consommer les API avec et sans streaming. Mesurer premier token, tokens/s et temps total à froid puis à chaud. |
| 1 h | Consigner le résultat dans un tableau et comparer l’inférence CPU locale aux architectures GPU et services managés. |

### TP · Appel API et mesure

~~~bash
curl -fsS --max-time 300 http://127.0.0.1:11434/api/chat -H 'Content-Type: application/json' -d '{"model":"qwen3:4b","messages":[{"role":"user","content":"Explique en français le rôle d’une sonde readiness Kubernetes en trois points."}],"think":false,"stream":false,"options":{"temperature":0,"num_ctx":4096,"num_predict":128}}' | jq '{reponse: .message.content, chargement_ms: (.load_duration/1e6), total_ms: (.total_duration/1e6), tokens: .eval_count, tokens_s: (if .eval_duration > 0 then .eval_count*1e9/.eval_duration else null end)}'
~~~

Pour observer la réponse au fil des tokens, passe `"stream":true`, ajoute `-N` à `curl` et enlève le filtre `jq`. Le serveur émet une suite d’objets JSON par ligne. Mesure le **TTFT** (*time to first token*) à partir du début de la requête jusqu’au premier fragment non vide de `message.content`.

~~~bash
python3 - qwen3:4b <<'PY'
import json
import sys
from time import perf_counter
from urllib.request import Request, urlopen

payload = {
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": "Explique en français le rôle d’une sonde readiness Kubernetes en trois points."}],
    "think": False,
    "stream": True,
    "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 128},
}
request = Request("http://127.0.0.1:11434/api/chat", json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"})
start = perf_counter()
first = None
with urlopen(request, timeout=240) as response:
    for line in response:
        update = json.loads(line)
        if first is None and update.get("message", {}).get("content"):
            first = round((perf_counter() - start) * 1_000)
        if update.get("done"):
            rate = update.get("eval_count", 0) * 1e9 / max(update.get("eval_duration", 0), 1)
            print({"ttft_ms": first, "total_ms": round((perf_counter() - start) * 1_000),
                   "tokens_s": round(rate, 2)})
PY
~~~

Répète en déchargeant le modèle avec `sudo docker compose exec ollama ollama stop qwen3:4b` pour une mesure à froid, puis immédiatement après pour une mesure à chaud. Note aussi le contexte, la version, la mémoire et les autres charges. Fais la même mesure avec 2 048 puis 4 096 tokens de contexte. Pour l’exercice JSON, demande les champs `titre`, `etapes` et `incertitudes`, impose le schéma via le champ `format` et valide la réponse avec Pydantic. [API Chat Ollama](https://docs.ollama.com/api/chat), [sorties structurées Ollama](https://docs.ollama.com/capabilities/structured-outputs).

**Livrable J1.** Accès distant confirmé, Modelfile versionné, un appel JSON valide et quatre mesures documentées au minimum.

## J2 · Construire et évaluer un RAG · 7 h

| Durée | Travail et résultat observé |
| --- | --- |
| 1 h 30 | Ingestion de Markdown, TXT et PDF texte. Conserver le fichier, la page et le numéro de fragment. |
| 1 h | Générer des embeddings BGE-M3, observer la dimension et indexer dans Qdrant. |
| 1 h | Examiner les résultats de `/search`. Comparer au moins deux découpages et regarder les erreurs de recherche. |
| 1 h 30 | Interroger `/ask`. Relier extraits, prompt, réponse structurée et citations. |
| 1 h 30 | Lancer l’évaluation, ajouter des questions reformulées et des questions sans réponse, comparer deux configurations. |
| 30 min | Présenter le flux de bout en bout et expliquer un échec avec sa trace ou ses mesures. |

### TP · Établir la référence puis changer une variable

1. Indexe le corpus et exécute les trois appels du [guide RAG](README.md#4-indexer-les-runbooks-et-poser-une-question). Lis [rag.py](rag.py) et localise le découpage, l’appel à `/api/embed`, la requête Qdrant et le prompt.
2. Lance `eval_lab.py`. Pour chaque question connue, relève **Hit@1**, **Hit@3**, la réponse et le fichier cité. Pour les questions hors corpus, relève les abstentions justifiées. Avec six documents et trois résultats, la recherche reste un exercice de base. Ajoute plusieurs documents distracteurs et au moins deux runbooks longs avant de juger le découpage.
3. Compare le découpage de référence (850 caractères avec 100 de recouvrement) à 450 caractères avec 80 de recouvrement. Réindexe avec `sudo docker compose --profile rag exec -T rag python rag.py --chunk-size 450 --overlap 80`. Ne change ni le modèle ni les questions pendant la comparaison.
4. Avec la configuration retenue, compare le chat direct et le RAG sur deux informations propres au lab Aster. Vérifie la source et le contenu cité. Ajoute une question dont la bonne réponse requiert deux passages et examine pourquoi elle est plus difficile.
5. Prépare un jeu de référence d’au moins 20 questions dont cinq sans réponse. Modifie [evaluation.json](evaluation.json), puis reconstruis le service avec `sudo docker compose --profile rag up -d --build rag` avant de relancer l’évaluation. Contrôle manuellement les citations. Les mots-clés du script repèrent des régressions simples, tandis que la vérification humaine teste le sens.

Si tu veux comparer le 8B, télécharge-le avec `sudo docker compose exec ollama ollama pull qwen3:8b`, ajoute une seule ligne `CHAT_MODEL=qwen3:8b` dans `.env`, puis lance `sudo docker compose --profile rag up -d --no-deps rag`. Garde le même index. Pour revenir au 4B, remplace la valeur de `CHAT_MODEL` par `qwen3:4b` et relance la même commande. Avant les mesures à chaud, envoie une question d’amorçage pour chaque modèle.

**Livrable J2.** Une API avec passages citables, un rapport reproductible et une décision 4B ou 8B motivée par qualité, mémoire et latence. Sur le jeu élargi, vise au départ au moins 80 % de Hit@3 pour les questions connues et vérifie manuellement les cinq abstentions. Le P95 d’un petit jeu est indicatif.

### Prolongement RAG

Compare une recherche par mots-clés et une recherche vectorielle sur un numéro de ticket et une question paraphrasée. Pour aller vers l’hybride, combine les classements puis mesure. Pour expérimenter LangChain, compose un `RunnableLambda` pour le retriever et une chaîne `prompt | ChatOllama | parser`, en gardant le même jeu de référence. Ajoute un reranking ou une réécriture des questions seulement si tu as identifié les cas que cette étape doit améliorer. [Recherche hybride Qdrant](https://qdrant.tech/documentation/concepts/hybrid-queries/), [intégration ChatOllama](https://docs.langchain.com/oss/python/integrations/chat/ollama).

## J3 · Intégrer des outils et exploiter le service · 7 h

| Durée | Travail et résultat observé |
| --- | --- |
| 1 h | Function calling. Le modèle propose un calcul et le code accepte uniquement l’outil et les arguments attendus. |
| 1 h | MCP. Démarrer un serveur STDIO, découvrir deux outils et deux ressources et appeler la recherche RAG. |
| 1 h 30 | LangGraph. Faire circuler un état typé, interrompre le workflow pour validation, puis tester les branches accord et refus. |
| 1 h | Sécurité. Tester une consigne parasite dans un document fictif, vérifier les citations et les refus. |
| 1 h 30 | LLMOps. Suivre les erreurs et les temps par étape, définir une régression et tester arrêt, reprise et accès privé. |
| 1 h | Dessiner le déploiement cible, estimer le coût et comparer un workflow unique à un système multi-agents. |

### TP · Outils, MCP et LangGraph

Le RAG doit être prêt sur `http://127.0.0.1:8001/readyz`. Installe les dépendances de ces exercices dans un environnement Python séparé sur la VM.

~~~bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install 'pydantic==2.14.3' 'pypdf==6.21.4' 'mcp>=1.26,<2' 'langgraph>=1.0.10,<2'
python -m ateliers.outils
python -m ateliers.serveur_mcp --test
python -m ateliers.workflow "Quand se fait la sauvegarde complète du lab Aster ?"
~~~

- **[Function calling](ateliers/outils.py).** Vérifie que l’outil calcule 135 minutes, que le nom est autorisé par le code et que `Duree` rejette `minutes=90`. Si Qwen3 4B omet l’appel, après téléchargement réessaie avec `CHAT_MODEL=qwen3:8b python -m ateliers.outils`. Mesure la latence des deux appels. [Appel d’outils dans Ollama](https://docs.ollama.com/capabilities/tool-calling).
- **[MCP](ateliers/serveur_mcp.py).** Le client de test lance le serveur local, liste ses capacités, calcule la durée et interroge `/search`. Le serveur expose des ressources prédéfinies. Il ne reçoit aucun chemin de fichier fourni par le modèle. [SDK MCP utilisé par l’exercice](https://pypi.org/project/mcp/1.26.0/).
- **[LangGraph](ateliers/workflow.py).** Essaie la même question avec `oui` puis `non`. Essaie une question hors corpus. Observe la reprise au point d’interruption avec le même `thread_id`. Le workflow affiche une réponse validée dans le terminal. Le checkpoint de cet exercice est en mémoire. [Interruptions LangGraph](https://docs.langchain.com/oss/python/langgraph/interrupts).

### TP · Recette d’exploitation

1. Ajoute à un document fictif une phrase sans rapport qui demande de répondre par un mot arbitraire. Réindexe, pose la question métier et vérifie que la phrase n’a pas orienté la réponse. Contrôle que le bon passage justifie la réponse. Retire la phrase, réindexe et compare.
2. Observe dans `/ask` les champs `retrieval_ms`, `generation_ms` et `output_tokens`. Ajoute des spans OpenTelemetry autour de l’embedding, de Qdrant et de la génération. Garde dans la trace la durée, la version et le statut. Évite d’y inclure les prompts et les runbooks. [Instrumentation Python OpenTelemetry](https://opentelemetry.io/docs/languages/python/instrumentation/).
3. Écris les critères d’un pipeline CI. Exécute les tests de schéma et de citations avec des réponses simulées, puis l’évaluation du modèle dans un job disposant de la machine et du corpus de test. Les questions sensibles et les identifiants restent dans le lab. En complément, teste DeepEval ou le jugement par un LLM local sur quelques exemples annotés, puis compare ses scores à ta lecture.
4. À l’aide des mesures, définis combien de requêtes simultanées la machine peut traiter et à partir de quel temps d’attente une file ou une mise à l’échelle devient nécessaire. Commence avec un utilisateur. Compare un contexte plus court, une réponse plafonnée et un modèle plus gros. Pour une application partagée, prévois authentification, contrôle d’accès aux documents, limitation de débit et invalidation du cache lors d’un changement d’index ou de droits.

**Livrable J3.** Démonstration depuis un autre réseau, appel d’outil validé, serveur MCP en lecture seule, workflow avec décision humaine, mesures de latence et procédure de reprise.

## Correspondance avec le programme fourni

| Modules | Mise en pratique |
| --- | --- |
| 1 à 5 · LLM, hébergement, prompts, outils et API | J1 pour l’hébergement, le streaming et la quantification. J3 pour le schéma et la validation des outils. |
| 6 à 7 · RAG et LangChain | J2 pour le RAG vectoriel en Python. Chaîne LCEL, recherche hybride et reranking en prolongement. |
| 8 à 10 · LangGraph, MCP et multi-agents | J3 pour un workflow et un serveur STDIO. En prolongement, comparer ce workflow à deux rôles séquentiels, recherche puis vérification, avec LangGraph ou Google ADK. Mesurer le coût en appels, les erreurs et la latence. |
| 11 à 13 · Évaluation, observabilité et sécurité | J2 et J3 pour le jeu de référence, le contrôle des sources, la validation des entrées, les temps de traitement et l’exercice OpenTelemetry. |
| 14 à 15 · Déploiement et montée en charge | J3 pour l’architecture en couches et les mesures sur CPU. En prolongement, placer l’API derrière une passerelle, puis comparer un runtime GPU avec Vertex AI, Bedrock ou Azure à partir d’un corpus fictif et du même protocole. |

Pour ton expérience AWS/EKS, la transposition la plus directe consiste à déployer l’API et son observabilité sur Kubernetes tout en laissant Ollama sur le mini-PC. En comparant ensuite une inférence GPU ou managée, mesure les mêmes indicateurs et documente la frontière de données. Pour le local, relève la puissance moyenne et les heures de fonctionnement. Les kWh correspondent à la puissance en watts multipliée par les heures et divisée par 1 000. Pour le managé, relève les tokens d’entrée et de sortie et vérifie les coûts d’hébergement et d’indexation.
