from src.preprocessing.oracle_tokenizer import tokenize_oracle_text


def test_tokenize_mana_and_numbers():
    tokens = tokenize_oracle_text("Draw a card.\nAdd {G}. Pay 2 life.")
    assert "draw_a_card" in tokens
    assert "MANA:{G}" in tokens
    assert "NUM:2" in tokens
