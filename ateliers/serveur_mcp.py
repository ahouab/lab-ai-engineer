import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp import FastMCP

from ateliers.outils import duree_minutes
from rag import Question, request_json


API = os.getenv("RAG_URL", "http://127.0.0.1:8001")
app = FastMCP("Runbooks du lab")


@app.tool()
def convertir_duree(heures: int, minutes: int) -> int:
    """Convertit une durée en minutes."""
    return duree_minutes(heures, minutes)


@app.tool()
def chercher_runbook(question: str) -> str:
    """Cherche les passages pertinents des runbooks du lab."""
    payload = Question(question=question).model_dump()
    return json.dumps(request_json(f"{API}/search", "POST", payload), ensure_ascii=False)


@app.resource("lab://config")
def configuration() -> str:
    return "Corpus de démonstration Aster. Deux outils en lecture seule."


@app.resource("lab://exemple/sauvegarde")
def sauvegarde() -> str:
    source = Path(__file__).resolve().parents[1] / "corpus" / "sauvegarde.md"
    return source.read_text(encoding="utf-8")


async def tester():
    server = StdioServerParameters(
        command=sys.executable, args=["-m", "ateliers.serveur_mcp"], env={"RAG_URL": API}
    )
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            outils = await session.list_tools()
            ressources = await session.list_resources()
            print("Outils", [item.name for item in outils.tools])
            print("Ressources", [str(item.uri) for item in ressources.resources])
            calcul = await session.call_tool("convertir_duree", {"heures": 2, "minutes": 15})
            recherche = await session.call_tool(
                "chercher_runbook", {"question": "À quelle heure se fait la sauvegarde complète ?"}
            )
            if calcul.isError or recherche.isError:
                raise RuntimeError(f"Appel MCP en erreur. {calcul.content} {recherche.content}")
            print("Calcul", calcul.content[0].text)
            print("Recherche", recherche.content[0].text)


if __name__ == "__main__":
    if "--test" in sys.argv:
        asyncio.run(tester())
    else:
        app.run(transport="stdio")
