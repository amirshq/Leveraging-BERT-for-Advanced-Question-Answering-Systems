def load_text(id_,root=""):
    with open(os.path.join(root,id_ + ".json")) as f: 
        text = json.load(f)
    return text
def clean_text(txt):
    return re.sub('[^A-Za-z0-9]+',' ',str(txt).lower()).strip()
def create_data():
    new_df = []
    for idx in tqdm(range(len(df))):
        article = load_text(df['Id'][idx],DATA_PATH_TRAIN)
        id_,pub_title,dataset_title,dataset_label, cleaned_label = df.iloc[idx]
        
        for i,section in enumerate(article):
            text = section['text']
            title = section['section_title']
            cleaned_text = clean_text(section['text'])
            found = cleaned_label in cleaned_text
            dic = {
                "id": [id_], 
                "section_id": [i],
                "pub_title": [pub_title], 
                "dataset_title": [dataset_title], 
                "dataset_label": [dataset_label], 
                "cleaned_label": [cleaned_label],
                "text": [text],
                "cleaned_text": [cleaned_text],
                "label_found": [found],
            }
            new_df.append(pd.DataFrame.from_dict(dic))
    return pd.concat(new_df).reset_index(drop=True)

df = pd.read_csv(DATA_PATH + 'train.csv')
new_df = create_data()
print(df)

df = new_df[new_df['label_found']].reset_index(drop=True)
df['lenght']= df['cleaned_text'].apply(lambda x: len(x.split()))
df = df[df['lenght'] < 3000]
df.to_csv("df_train.csv",index=False)