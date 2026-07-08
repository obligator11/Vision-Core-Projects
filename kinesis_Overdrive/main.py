"""
Project Kinesis: NEXT
Entry point. Run with: python main.py
"""

from engine.app import KinesisApp


def main():
    app = KinesisApp()
    app.run()


if __name__ == "__main__":
    main()
