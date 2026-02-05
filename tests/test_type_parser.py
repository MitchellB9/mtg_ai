from src.preprocessing.type_parser import parse_type_line


def test_parse_simple_creature():
    p = parse_type_line("Legendary Creature — Human Soldier")
    assert "Creature" in p.basic_types
    assert "Legendary" in p.super_types
    assert "Human" in p.sub_types
    assert "Soldier" in p.sub_types


def test_parse_split_faces():
    p = parse_type_line("Creature — Elf // Sorcery")
    assert len(p.faces) == 2
