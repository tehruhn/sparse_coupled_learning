a = "A man, nam a"

print("".join(ch for ch in a.lower() if ch.isalpha()) == "".join(ch for ch in a.lower() if ch.isalpha())[::-1])