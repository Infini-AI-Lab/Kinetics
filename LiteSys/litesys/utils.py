class TextColors:

    COLORS = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "reset": "\033[0m",
    }

    @staticmethod
    def colorize(text, color):

        color_code = TextColors.COLORS.get(color.lower(), TextColors.COLORS["reset"])
        return f"{color_code}{text}{TextColors.COLORS['reset']}"
