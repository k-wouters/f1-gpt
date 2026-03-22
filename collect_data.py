"""
Collects Formula 1 text from Wikipedia and saves it as a single training corpus.
"""

import os
import wikipediaapi

# F1 Wikipedia articles to collect
F1_ARTICLES = [
    # History & overview
    "Formula One",
    "History of Formula One",
    "Formula One regulations",
    "Formula One car",

    # Iconic drivers
    "Michael Schumacher",
    "Ayrton Senna",
    "Lewis Hamilton",
    "Max Verstappen",
    "Alain Prost",
    "Niki Lauda",
    "Juan Manuel Fangio",
    "Sebastian Vettel",
    "Fernando Alonso",
    "Mika Häkkinen",
    "Nigel Mansell",
    "Jackie Stewart",
    "Jim Clark",
    "Kimi Räikkönen",
    "Nelson Piquet",
    "Jenson Button",

    # Teams
    "Scuderia Ferrari",
    "McLaren",
    "Mercedes-Benz in Formula One",
    "Red Bull Racing",
    "Williams Racing",
    "Renault in Formula One",
    "Lotus F1 Team",
    "Brabham",

    # Circuits
    "Monaco Grand Prix",
    "Italian Grand Prix",
    "British Grand Prix",
    "Belgian Grand Prix",
    "Japanese Grand Prix",
    "Brazilian Grand Prix",
    "Circuit de Monaco",
    "Silverstone Circuit",
    "Monza circuit",
    "Spa-Francorchamps",

    # Seasons
    "1994 Formula One World Championship",
    "2021 Formula One World Championship",
    "2016 Formula One World Championship",
    "2008 Formula One World Championship",
]


def fetch_article(wiki, title):
    page = wiki.page(title)
    if page.exists():
        print(f"  Fetched: {title} ({len(page.text)} chars)")
        return page.text
    else:
        print(f"  Skipped (not found): {title}")
        return ""


def main():
    os.makedirs("data", exist_ok=True)
    output_path = "data/f1_corpus.txt"

    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="f1-gpt/1.0 (educational project)"
    )

    print("Collecting F1 Wikipedia articles...")
    all_text = []

    for title in F1_ARTICLES:
        text = fetch_article(wiki, title)
        if text:
            all_text.append(f"\n\n=== {title} ===\n\n{text}")

    corpus = "\n".join(all_text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(corpus)

    char_count = len(corpus)
    word_count = len(corpus.split())
    print(f"\nDone! Saved to {output_path}")
    print(f"Total: {char_count:,} characters, ~{word_count:,} words")


if __name__ == "__main__":
    main()
