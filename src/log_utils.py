from math import floor, ceil


def log_title(title, title_length=50, padding_char="="):
    title = f" {title} "
    padding_length = title_length - len(title)
    left_padding = floor(padding_length / 2)
    right_padding = ceil(padding_length / 2)
    print(f"{padding_char * left_padding}{title}{padding_char * right_padding}")
