import argparse
import sys
from pipeline import Pipeline


def interactive_menu():
    print("\n" + "="*60)
    print("   🎙  MULTILINGUAL SPEECH-TO-SUMMARY PIPELINE")
    print("="*60)
    print("\nSelect input mode:")
    print("  1. Record from microphone")
    print("  2. Load from audio file")
    print("  3. Enter text directly")
    print("  4. Model comparison")
    print("  5. Exit")

    choice = input("\nChoice [1-5]: ").strip()
    if choice == "5":
        sys.exit(0)

    language = None
    if choice in ("1", "2", "4"):
        print("\nLanguage:\n  1. Auto-detect\n  2. English\n  3. Malayalam\n  4. Hindi\n  5. Other")
        lang_choice = input("\nLanguage [1-5, default=1]: ").strip() or "1"
        lang_map    = {"2": "en", "3": "ml", "4": "hi"}
        if lang_choice in lang_map:
            language = lang_map[lang_choice]
        elif lang_choice == "5":
            language = input("Enter ISO code: ").strip()

    print("\nSummarizer:\n  1. T5-small (fast)\n  2. DistilBART (better)\n  3. BART-large (best)")
    model_choice = input("\nModel [1-3, default=1]: ").strip() or "1"
    model        = {"1": "t5-small", "2": "distilbart", "3": "bart-large"}.get(model_choice, "t5-small")

    pipeline = Pipeline(whisper_model="small", summarizer_model=model)

    if choice == "1":
        try:
            duration = int(input("\nDuration in seconds [default=30]: ").strip() or "30")
        except ValueError:
            duration = 30
        pipeline.run_from_mic(duration=duration, language=language)

    elif choice == "2":
        file_path = input("\nAudio file path: ").strip().strip('"')
        pipeline.run_from_file(file_path=file_path, language=language)

    elif choice == "3":
        print("\nPaste text (blank line to finish):")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        pipeline.run_from_text(text=" ".join(lines))

    elif choice == "4":
        file_path = input("\nAudio file for comparison: ").strip().strip('"')
        pipeline.run_comparison(audio_path=file_path, language=language)


def cli_main():
    parser = argparse.ArgumentParser(description="Multilingual Speech-to-Summary Pipeline")
    parser.add_argument("--file",     type=str)
    parser.add_argument("--mic",      action="store_true")
    parser.add_argument("--text",     type=str)
    parser.add_argument("--compare",  action="store_true")
    parser.add_argument("--lang",     type=str, default=None)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--model",    type=str, default="t5-small", choices=["t5-small", "distilbart", "bart-large"])
    parser.add_argument("--whisper",  type=str, default="small",    choices=["tiny", "base", "small"])
    args = parser.parse_args()

    if not any([args.file, args.mic, args.text, args.compare]):
        interactive_menu()
        return

    pipeline = Pipeline(whisper_model=args.whisper, summarizer_model=args.model)

    if args.compare:
        if not args.file:
            print("❌  --compare requires --file")
            sys.exit(1)
        pipeline.run_comparison(args.file, language=args.lang)
    elif args.mic:
        pipeline.run_from_mic(duration=args.duration, language=args.lang)
    elif args.file:
        pipeline.run_from_file(args.file, language=args.lang)
    elif args.text:
        pipeline.run_from_text(args.text)


if __name__ == "__main__":
    cli_main()
