Files

final.edgelist: Edgelist of the graph. Nodes are indexed as user ids and edges are followship between users.

vax_final.json: dict with user ids as keys and list of tweet info as values. Each list contains information of tweets of each user that are has hashtag related to vaccination. The information includes the datetime, tweet id, and subjectivity together with polarity scores from the nltk package.

vax_gpt_labels.pkl: dict with tweet ids as keys and integer scores as values. Scores are returned by gpt3.5-turbo between 0 and 10, where 0 means strongly against vaccination and 10 means fully support it. Only tweets with valid scores will appear in this dict.

war_final.json and war_gpt_labels.pkl are similar but focus on tweets with hashtags related to the ukraine war.

user_id_dict.pkl: dict with user ids as keys and user info as values.