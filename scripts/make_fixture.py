from pathlib import Path
import pandas as pd
rows=[]
examples=[
("My package has not arrived", "Please DM your order number so we can check the delivery status."),
("Where is my delivery?", "Please DM your order number so we can check the delivery status."),
("I need a refund", "We can review your order and refund eligibility in a secure DM."),
("Please refund my purchase", "We can review your order and refund eligibility in a secure DM."),
("I was charged twice", "Please DM us securely with the transaction details for review."),
("Why is there an extra charge?", "Please DM us securely with the transaction details for review."),
("I cannot log in", "Please use the password reset flow; a specialist can help if it continues."),
("My account is locked", "Please use the password reset flow; a specialist can help if it continues."),
("How do I cancel?", "Please send us a DM and we will review your cancellation request."),
("Cancel my subscription", "Please send us a DM and we will review your cancellation request."),
("The app is down", "We are checking the service issue and will share updates here."),
("Your site is not working", "We are checking the service issue and will share updates here."),
]
for i,(customer,reply) in enumerate(examples):
    root=str(1000+i); rows.extend([
      {"tweet_id":root,"author_id":f"user{i}","inbound":True,"created_at":"2017-01-01","text":customer,"response_tweet_id":str(2000+i),"in_response_to_tweet_id":None},
      {"tweet_id":str(2000+i),"author_id":"AcmeSupport","inbound":False,"created_at":"2017-01-01","text":reply,"response_tweet_id":None,"in_response_to_tweet_id":root},])
path=Path("data"); path.mkdir(exist_ok=True); pd.DataFrame(rows).to_csv(path/"fixture_tweets.csv",index=False)
print("Wrote data/fixture_tweets.csv; fixture only, not a submission dataset.")
