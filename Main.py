from sklearn.model_selection import train_test_split
from DataSource import load_normalized_data, load_normalized_data_only_regex, load_data
from Clasyfication import tfidf_parameters, SVC_VS_NB, SVC_paramteres, selector_chi, \
    multinominal_nb_alpha_test, no_text_cleaning, cv_parameter_1
from TextClasyfication import category_analyze, data_selection
from TextPreprocesing import text_cleaning_only_regex, text_cleaning

selected_categories = ["Fantasy", "Novel", "Children's literature", "Science Fiction"]


def split_df_to_train_test(df):
    return train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        stratify=df['label'],
        shuffle=True)

df = load_data()
category_analyze(df)
selected_data = data_selection(df, selected_categories)
print(selected_data)
text_cleaning_only_regex(selected_data, selected_categories)
text_cleaning(selected_data, selected_categories)


d_farme = load_normalized_data()
X_train, X_Test, y_train, y_test = split_df_to_train_test(d_farme)
tfidf_parameters(X_train, X_Test, y_train, y_test, selected_categories)
SVC_VS_NB(X_train, X_Test, y_train, y_test, selected_categories)
SVC_paramteres(X_train, X_Test, y_train, y_test, selected_categories)
selector_chi(X_train, X_Test, y_train, y_test, selected_categories)
multinominal_nb_alpha_test(X_train, X_Test, y_train, y_test, selected_categories)
cv_parameter_1(X_train, X_Test, y_train, y_test, selected_categories)
X_train, X_Test, y_train, y_test = split_df_to_train_test(load_normalized_data_only_regex())
no_text_cleaning(X_train, X_Test, y_train, y_test, selected_categories)
