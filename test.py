def reverseWords(s: str) -> str:
    words = s.split(" ")

    for i in range(len(words)):
        words[i] = words[i][::-1]

    print(words)
    res = " ".join(words)
    return res

reverseWords('hiang dep trai')