import pandas as pd

df = pd.read_csv("naive.csv")
age = df.groupby(['age','buys']).size()
income = df.groupby(['income','buys']).size()
student = df.groupby(['student','buys']).size()
credit = df.groupby(['credit','buys']).size()

collection = [age,income,student,credit]

total_yes = df['buys'].value_counts()['yes']
total_no = df['buys'].value_counts()['no']
total = len(df)

prior_yes = total_yes / total
prior_no = total_no / total
input_data = {'age':'youth','income':'medium','student':'yes','credit':'fair'}

def naive_classify(data):
    yes = prior_yes
    no = prior_no
    for i,j in zip(data.values(),collection):    
        yes *= j.get((i,'yes'),0)/total_yes
        no *= j.get((i,'no'),0)/total_no
         
    return float(yes),float(no)

yes,no = naive_classify(input_data)
print(input_data)
print(f"P(Yes|x): {yes}")
print(f"P(No|X): {no}")
if yes>no:
    print("Yes has more probability")
else:
    print("No has more probability")
