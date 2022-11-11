def cos_similarity(x, y, nlp):
    """ return cosine similarity between two lists """
    x_embed = nlp(str(x)).vector
    y_embed = nlp(str(y)).vector

    numerator = sum(a * b for a, b in zip(x_embed, y_embed))
    squared_sum_x = sum(i * i for i in x_embed)
    squared_sum_y = sum(i * i for i in y_embed)

    denominator = squared_sum_x * squared_sum_y
    return round(numerator / float(denominator), 3)


def find_similarity(first_set, sec_set, tech_skills, regular_skills, nlp):
    sums = 0.0
    for fs in first_set:
        for ss in sec_set:
            sums += cos_similarity(str(fs), str(ss), nlp)

            # If they both in tech skills, reward.
            if (str(fs) in tech_skills) and (str(ss) in tech_skills):
                # print("tech")
                sums += 0.1

            # If they both in regular skills, reward.
            if (str(fs) in regular_skills) and (str(ss) in regular_skills):
                # print("regular")
                sums += 0.1
    return sums
