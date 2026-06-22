"""Intelligence Layer: agent cards compose prompt + skills + tools + memory."""

from erdos_fai import AgentCard, Memory, Skill, Tool


def test_render_system_includes_all_parts():
    card = AgentCard(
        name="Analyst",
        system="You are a data analyst.",
        skills=[Skill("charts", "Always describe trends.")],
        tools=[Tool("query_db", "Run a read-only SQL query", handler=lambda **k: None)],
    )
    card.memory.remember("Prefer ISO-8601 dates.")
    rendered = card.render_system()

    assert "You are a data analyst." in rendered
    assert "Skill: charts" in rendered
    assert "query_db" in rendered
    assert "ISO-8601" in rendered


def test_write_tool_flag_and_catalog():
    send = Tool("send_email", "Send an email", handler=lambda **k: "sent", write=True)
    card = AgentCard(name="Mailer", tools=[send])
    assert card.tools.get("send_email").write is True
    assert "(write)" in card.tools.catalog()


def test_memory_recall_and_context():
    mem = Memory()
    mem.remember("a", tag="x")
    mem.remember("b", tag="y")
    assert len(mem.recall()) == 2
    assert len(mem.recall(tag="x")) == 1
    assert "a" in mem.as_context()
