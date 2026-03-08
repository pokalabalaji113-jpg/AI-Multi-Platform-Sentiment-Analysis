def generate_reply(sentiment):

    if sentiment == "Positive":
        return "Thank you for your positive feedback! We appreciate your support."

    elif sentiment == "Negative":
        return "We are sorry for the inconvenience. Our team will work to improve your experience."

    else:
        return "Thank you for sharing your opinion."