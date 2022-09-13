def prettyFormat(text):
    """If text is of the form <http://purl.obolibrary.org/obo/GO_0071554> this function \
        returns GO:0071554

    :param text: Text to be formatted
    :type text: str

    :rtype: str
    """
    if text[0] == "<" and text[-1] == ">":
        text = text[1:-1]
        text = text.split("/")[-1]
        text = text.replace("_", ":")
    elif text.startswith("http"):
        text = text.split("/")[-1]
        text = text.replace("_", ":")
    elif text.startswith("GO:"):
        pass
    else:
        pass
    #            raise Exception("prettyFormat: unrecognized text format: %s", text)
    return text
