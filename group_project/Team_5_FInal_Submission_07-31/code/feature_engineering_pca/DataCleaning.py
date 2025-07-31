#<-- Begin Requirements -->#
'''
pandas
numpy
scikit-learn
urlextract
list
torch
transformers
'''
#<-- End Requirements --#


#<-- Begin Class Definition -->#

#<-- Begin Embedding Extraction -->#
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import copy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#<-- END -->#

class embedding_extraction():
    '''
    Build to assist in embedding extraction.
    Currently limited to mean-pooling with DistilBERT
    '''
    def __init__(self, language_model='distilbert', desired_device='GPU'):
        '''
        Currently, language_model is not used. It is here as a reminder to 
            generalize in the future.
        Initialize tokenizer and model for later use in embedding generation.
        If available, use GPU.
        Note: it would be a good idea to add a copy-level option later, 
            since deep can be expensive and unecessary sometimes.
        '''
        # Load tokenizer and model
        self.language_model = language_model
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
        self.model = AutoModel.from_pretrained("distilbert-base-cased")
        # Use GPU if available
        if desired_device == 'GPU':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.model = self.model.to(self.device)

    def mean_pooling(self, model_output, attention_mask):
        '''
        Generic mean_pooling funciton, inspired by https://www.byteplus.com/en/topic/496887?title=distilbert-get-embeddings-a-complete-guide
            Adjusted to sum across dim=0 (not dim=1, to maintain the 768 DistilBERT features)
            Added .squeeze(0) to fit ensure fitting dimensionality requirements
        '''
        token_embeddings = model_output.last_hidden_state.squeeze(0)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum = torch.sum(token_embeddings * input_mask_expanded, dim=0)  # Changed dim from 1 to 0
        count = torch.clamp(input_mask_expanded.sum(dim=0), min=1e-9)   # Changed dim from 1 to 0
        mean = (sum / count).cpu().numpy()
        return mean


    def get_mean_pooling(self, text):
        '''
        Generic mean_pooling fetcher, inspired by https://www.byteplus.com/en/topic/496887?title=distilbert-get-embeddings-a-complete-guide
            Added .to(device) line to ensure GPU compatibility (if available)
            Added .squeeze(0) to fit ensure fitting dimensionality requirements
        '''
        # Tokenize sentences
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling. In this case, mean pooling.
        text_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'].squeeze(0))
        return text_embeddings
    
    def get_embeddings(self, input_df, input_column, pool_type='mean'):
        '''
        In current implementation, pool_type is unused.
            It is present so future iterations may be update for generality
                (allowing different pooling methods)
        This function generates mean-pooled DistilBERT embeddings and appends 
            them to the input dataframe.
        '''
        # Copy df
        df = copy.deepcopy(input_df)

        # Get Embeddings
        df["embedding"] = df[input_column].apply(self.get_mean_pooling)

        embedding_df = pd.DataFrame(df["embedding"].tolist(), index=df.index)
        embedding_df.columns = [f"emb_{i}" for i in embedding_df.columns]

        df1 = pd.concat([df, embedding_df], axis=1)
        df1 = df1.drop(columns=["embedding"])
        return df1
    
    def replace_text_with_embeddings(self, input_df, input_column, pool_type='mean'):
        '''
        Replaces the input_column with it's textual embeddings.
            Currently uses only DistilBERT to generate those embeddings.
        '''
        df = copy.deepcopy(input_df)
        df1 = self.get_embeddings(df, input_column, pool_type)
        df2 = df1.drop(columns=[input_column])
        return df2

    def get_embeddings_with_pca(self, input_df, input_column, pool_type='mean', pca=None, scaler=None, target_variance=0.95):
        '''
        In current implementation, pool_type is unused.
            It is present so future iterations may be update for generality
                (allowing different pooling methods)
        This function generates mean-pooled DistilBERT embeddings and appends 
            them to the input dataframe.
        variable "pca" should contain a pca model for input, or nothing.
            If pca=None, pca is defined and utilized
            If pca=pca_model, the provided model is used
        variable "scaler" is similar to PCA. 
        Inspired by https://mikulskibartosz.name/pca-how-to-choose-the-number-of-components
            and https://www.byteplus.com/en/topic/496887?title=distilbert-get-embeddings-a-complete-guide
        '''
        if pool_type != 'mean':
            raise NotImplementedError("Only 'mean' pooling is currently implemented.")

        # Copy df
        df = copy.deepcopy(input_df)

        # Get Embeddings
        df["embedding"] = df[input_column].apply(self.get_mean_pooling)

        embedding_df = pd.DataFrame(df["embedding"].tolist(), index=df.index)
        embedding_df.columns = [f"emb_{i}" for i in embedding_df.columns]

        if pca is None:
            # Define and use PCA
            pca = PCA(n_components=target_variance)
            scaler = StandardScaler()
            scaler.fit(embedding_df)
            embedding_scaled_df = scaler.transform(embedding_df)
            pca.fit(embedding_scaled_df)
            embedding_reduced_df = pca.transform(embedding_scaled_df)
            embedding_reduced_df = pd.DataFrame(embedding_reduced_df, index=df.index)
            embedding_reduced_df.columns = [f'pca_{i}' for i in embedding_reduced_df.columns]

        else:
            if scaler is None:
                raise ValueError("Please provide a scaler.")
            embedding_scaled_df = scaler.transform(embedding_df)
            embedding_reduced_df = pca.transform(embedding_scaled_df)
            embedding_reduced_df = pd.DataFrame(embedding_reduced_df, index=df.index)
            embedding_reduced_df.columns = [f'pca_{i}' for i in embedding_reduced_df.columns]
            # Use provided PCA

        df1 = pd.concat([df, embedding_reduced_df], axis=1)
        df1 = df1.drop(columns=["embedding"])
        return df1, pca, scaler
    
    def replace_text_with_embeddings_with_pca(self, input_df, input_column, pool_type='mean', pca=None, scaler=None, target_variance=0.95):
        '''
        Replaces the input_column with it's textual embeddings.
            Currently uses only DistilBERT to generate those embeddings.
        '''
        df = copy.deepcopy(input_df)
        df1, pca_model, scaler_model = self.get_embeddings_with_pca(df, input_column, pool_type, pca, scaler, target_variance)
        df2 = df1.drop(columns=[input_column])
        return df2, pca_model, scaler_model
#<-- End Embedding Extraction --#>

#<-- Begin URL Extraction -->#
import re
from urlextract import URLExtract
from collections import Counter
import copy

class url_extraction():
    '''
    A collection of functions for extracting URLs from text.
    '''
    def __init__(self):
        self.urlextract = URLExtract()
        self.regex = re.compile(r'((?:(?:http|https|ftp):\/\/|www\.)(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-]))')
        self.TRAILING_CHARS = '.,!?;:()[]{}<>|^\\ \'"'

    def urlex_extractor(self, text):
        # Extraction with URLExtract
        return self.urlextract.find_urls(text)
    
    def regex_extractor(self, text):
        # Extraction with RegEx
        return self.regex.findall(text)
    
    def regex_urlex_extractor(self, text):
        # Extraction with RegEx, cleaned with URLExtract
        reg_url = self.regex_extractor(text)
        return [url for u in reg_url for url in self.urlex_extractor(u)]
    
    def regex_endchar_extractor(self, text):
        # Extraction with RegEx, cleaned with common trailing chars list
        reg_url = self.regex_extractor(text)
        return [u.rstrip(self.TRAILING_CHARS) for u in reg_url]

    def string_extractor(self, text, regex_cleaner='urlex'):
        '''
        Extracts all URLs (that match certain criteria) from a string.
        We use both URLExtract and RegEx because URLExtract checks URL's based on TLD
        regex_cleaner takes the following values:
        - urlex
        - endchar
        They call their respective regex_<regex_cleaner>_extractor functions
        '''
        url_list = []
        if regex_cleaner == 'urlex':
            url_list += self.regex_urlex_extractor(text)
        elif regex_cleaner == 'endchar':
            url_list += self.regex_endchar_extractor(text)
        else:
            raise ValueError('Invalid regex_cleaner. Please select either "urlex" or "endchar".')
        
        # Filter the urlex output to avoid false doubles 
            # (regex and urlex both counting the same urls)
        urlex_urls = self.urlex_extractor(text)
        filtered_urlex_urls = [url for url in urlex_urls if url not in url_list]
        url_list += filtered_urlex_urls

        return url_list
    
    def url_list_to_dict(self, url_list):
        '''
        Turns a given list of URLs to a dictionary of URL counts
        '''
        return dict(Counter(url_list))
    
    def string_to_dict(self, text):
        urls = self.string_extractor(text)
        return self.url_list_to_dict(urls)

    def list_extractor(self, lst):
        raise NotImplementedError("This method is not implemented yet.")
    
    def df_extractor(self, input_df, input_column, output_column):
        df = copy.deepcopy(input_df)
        df[output_column] = df[input_column].apply(self.string_to_dict)
        return df
#<-- End URL Extraction -->#

#<-- Begin URL Handling -->#
'''
Status Format:
lowercase for unresolved, unimplemented
uppercase for RESOLVED, IMPLEMENTED


----------------------------
Known Bugs:
----------------------------
1: Discovered 7/5/2025  (BugID 0001)
- If text contains something akin to:
    - yada yada google.com yada yada google.com/test
    and the dictionary contains
    - google.com: 1, google.com/test: 1
    The current implementation will do replacement as follows:
    - yada yada <url1> yada yada <url1>/test
    or
    - yada yada <url> yada yada <url>/test
    This is not the intended behavior. We want to see:
    - yada yada <url1> yada yada <url2>
    or
    - yada yada <url> yada yada <url>
Comments:
- For the purposes of raw replacement, this bug is problematic.
- For the purposes of machine learning from language embeddings
    (like, for example, from a BERT-based model), this may not be 
    so bad. The subsequent words may provide insight into what the 
    link is for that the model would otherwise miss.
    For example: <url2>/login may provide more context to the 
    generated tokens than <url3>
    - There is still one glaring issue though: consistency
        Sometimes, the order may be url, url/login.
        Other times, it may be      url/login, url.
        - When fixing, implement the fix suggested in FixID 0001.1, 
            then also implement it's inverse (order by decreasing value)
            - Perhaps better, make that a passable argument to the 
                fix function.
            This way, both methods can be utilized, and it can be discovered 
                which is better.
    - It is also worth pointing out that sometimes we won't have an example of 
        just the base url being present. It would be best to build a 
        base-extractor 
        - (for example, extracting example.com from example.com/login)
        prior to using the aformentioned "inverse" option.
Status:     #<-- RESOLVED -->#



----------------------------  
Suggested Fixes:
----------------------------
1: Suggested 7/5/2025   (BugID 0001, FixID 0001.1)
- Use a similar search method (as to what is used for our find/replace)
    to locate matching strings in the dictionary keys.
    - Then, alter the fill order to fill the non-contained strings first, then 
        proceed down the containment list.
    - To do this: 
        - Cycling through the url_dict, create a separate containment_dict of the 
            form {url: contained_in_count}, with 
            contained_in_count = # of other urls containing this url.
        - re-order containment_dict by increasing value.
            For example, {url1: 2, url2: 0, url3: 2}
                becomes  {url2: 0, url1: 2, url3: 2},
                or       {url1: 2, url2: 0, url3: 1}
                becomes  {url2: 0, url3: 1, url1: 2}
        - Then, find and replace in the order of containment_dict
            - To maintain the order of url_dict, prior to performing 
                find and replace on the text column from containment_dict, 
                perform find and replace on containment_dict from url_dict.
Status:     #<-- IMPLEMENTED -->#

'''



import re
import copy

class url_handling():
    '''
    A class storing functions for handling URLs in data cleaning.
    '''
    def __init__(self):
        pass
        # try:
        #     import URLExtraction
        # except ImportError:
        #     print('Failed to import URLExtraction. Please ensure the file is available.')



    def replace_urls_from_dict(self, text_col, url_dict_col, token_pre='url', indexed=False, inverse=False):
        '''
        Replaces URLs in a body of text using a dictionary of URLs.

        Parameters:
            - text_col: Name of text column
            - url_dict_col: Name of URL dictionary column
            - token_pre: Prefix for the token. A proceeding "<" and ending ">" 
                or "i>" is added later in the code, based on the value of 'indexed'
            - indexed: If True, uses numbered tokens; else, a single shared token
            - inverse: If True, most contained URLs first (decreasing containment)
        '''
        def replace(row):
            text = row[text_col]
            url_dict = row[url_dict_col]
            if not url_dict:
                return text

            url_list = list(url_dict.keys())  # Original order

            # Assign tokens based on original order
            if indexed:
                token_map = {url: f'<{token_pre}{i+1}>' for i, url in enumerate(url_list)}
            else:
                token_map = {url: f'<{token_pre}>' for url in url_list}

            # Compute containment count
            containment_count = {
                u: sum(1 for other in url_list if u != other and u in other)
                for u in url_list
            }

            # Sort URLs by containment count
            replacement_order = sorted(
                url_list,
                key=lambda u: containment_count[u],
                reverse=inverse
            )

            # Replace in text using sorted order and original token_map
            for url in replacement_order:
                token = token_map[url]
                text = re.sub(re.escape(url), token, text)

            return text

        return replace

    
    def url_replacement(self, input_df, text_col, url_dict_col, output_col, url_dtype='dict', token_pre='url', indexed=False, inverse=False):
        '''
        Replaces URLs.
        input_df: the dataframe to be altered
        text_col: the name of the column containing the text that we wish to substitute parts of
        url_dict_col: the name of the column containing the url dictionary
            currently only dictionaries are implemented: name may change if more implemented later
        output_col: the name of the column we wish to store the output in.
        url_dtype: represents the format in which the urls are stored. Options:
        - dict  (default)
        - No more options currently implemented.
        token_pre: the desired replacement value. See replace_urls_from_dict
        indexed: a Boolean variable.
        '''
        if url_dtype == 'dict':
            df = copy.deepcopy(input_df)
            df[output_col] = df.apply(self.replace_urls_from_dict(text_col, url_dict_col, token_pre=token_pre, indexed=indexed, inverse=inverse), axis=1)
            return df
        else:
            raise ValueError('Only url_dtype="dict" is currently implemented. Please double check the parameters you are passing.')
#<-- End URL Handling -->#

#<-- End Class Definition -->#


#<-- Begin Data Split -->#

def Data_Split():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from pathlib import Path

    # Grab inspected rows: they MUST be in the training set
    Nazario_aid = [0, 1]   # Manual. There was a mistake in 1_Code_Inspect, so a random sample was never taken
    Ling_aid = [1584, 1746, 772, 1729, 387, 407, 2441, 1992, 2448, 2755]


    # Import Data
    # 7-Column Sets
    Nigerian_Fraud_df = pd.read_csv("data/Nigerian_Fraud.csv")
    Nazario_df = pd.read_csv("data/Nazario.csv")
    SpamAssasin_df = pd.read_csv("data/SpamAssasin.csv")
    CEAS_08_df = pd.read_csv("data/CEAS_08.csv")

    # 3-Column Sets
    Ling_df = pd.read_csv("data/Ling.csv")
    Enron_df = pd.read_csv("data/Enron.csv")


    # Separate inspected rows
    # Analyzed rows
    Ling_analyzed = Ling_df.loc[Ling_aid]
    Nazario_analyzed = Nazario_df.loc[Nazario_aid]

    # Remaining rows
    Ling_remaining = Ling_df.drop(Ling_aid)
    Nazario_remaining = Nazario_df.drop(Nazario_aid)


    # Train/test split 1: results in testing set.
    # First Train/Test for 7-Colum no analysis
    Nigerian_Fraud_train1, Nigerian_Fraud_test = train_test_split(Nigerian_Fraud_df, test_size=0.2, random_state=42)
    SpamAssasin_train1, SpamAssasin_test = train_test_split(SpamAssasin_df, test_size=0.2, random_state=42)
    CEAS_08_train1, CEAS_08_test = train_test_split(CEAS_08_df, test_size=0.2, random_state=42)

    # First Train/Test for 7-Column with analysis
    Nazario_train1, Nazario_test = train_test_split(Nazario_remaining, test_size=0.2, random_state=42)

    # First Train/Test for 3-Colum no analysis
    Enron_train1, Enron_test = train_test_split(Enron_df, test_size=0.2, random_state=42)

    # First Train/Test for 3-Column with analysis
    Ling_train1, Ling_test = train_test_split(Ling_remaining, test_size=0.2, random_state=42)


    # Train/test split 2: results in training and validation sets.
    # Second Train/Test for 7-Colum no analysis
    Nigerian_Fraud_train, Nigerian_Fraud_val = train_test_split(Nigerian_Fraud_train1, test_size=0.2, random_state=42)
    SpamAssasin_train, SpamAssasin_val = train_test_split(SpamAssasin_train1, test_size=0.2, random_state=42)
    CEAS_08_train, CEAS_08_val = train_test_split(CEAS_08_train1, test_size=0.2, random_state=42)

    # Second Train/Test for 7-Column with analysis
    Nazario_train2, Nazario_val = train_test_split(Nazario_train1, test_size=0.2, random_state=42)
    Nazario_train = pd.concat([Nazario_train2, Nazario_analyzed])

    # Second Train/Test for 3-Colum no analysis
    Enron_train, Enron_val = train_test_split(Enron_train1, test_size=0.2, random_state=42)

    # Second Train/Test for 3-Column with analysis
    Ling_train2, Ling_val = train_test_split(Ling_train1, test_size=0.2, random_state=42)
    Ling_train = pd.concat([Ling_train2, Ling_analyzed])


    # Sanity check: no duplicates
    # Verify no duplicate indices
    try:
        Ling_train.index.has_duplicates
    except:
        raise IndexError("Duplicate indices found in Ling_train. Verify the code flow.")

    try:
        Nazario_train.index.has_duplicates
    except:
        raise IndexError("Duplicate indices found in Nazario_train. Verify the code flow.")


    # Output train/val/test sets for safe keeping.
    # Make output directory if it doesn't exist
    output_dir = Path("Split_Data/Uncleaned")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set output path to previously make output directory, save files
    to_save = ['Nigerian_Fraud', 'Nazario', 'SpamAssasin', 'CEAS_08', 'Enron', 'Ling']
    save_levels = ['_train', '_val', '_test']

    for dataset in to_save:
        for level in save_levels:
            var_name = f"{dataset}{level}"
            df = locals().get(var_name)

            if df is not None:
                output_path = output_dir / f"{var_name}.csv"
                df.to_csv(output_path, index=False)
                print(f"Saved {var_name} to {output_path}")
            else:
                print(f"Variable '{var_name}' not found.")

#<-- End Data Split -->#


#<-- Begin Feature Engineering -->#

def Feature_Engineering():
    import pandas as pd
    from pathlib import Path

    # Prep import lists
    from_save = ['Nigerian_Fraud', 'Nazario', 'SpamAssasin', 'CEAS_08', 'Enron', 'Ling']
    save_levels = ['_train', '_val', '_test']

    # Define list of dataframes
    df_train_list = []
    df_val_list = []
    df_test_list = []

    # Define necessary class instances
    extract_urls = url_extraction()
    replace_urls = url_handling()


    # Perform feature engineering
    # Make output directory if it doesn't exist
    input_dir = Path("Split_Data/Uncleaned")
    input_dir.mkdir(parents=True, exist_ok=True)

    for dataset in from_save:
        for level in save_levels:
            var_name = f"{dataset}{level}"
            input_path = input_dir / f"{var_name}.csv"
            df = pd.read_csv(input_path)
            if df is not None:
                '''
                Dropping all columns other than subject, body, and label.
                    This way, we can use all 6 datasets
                    The url extractor designed in RULExtraction will be used to 
                        handle URLs. We will have count and distinct_count, which 
                        will contain all information (and more) from urls
                            (urls was just a 0,1 column stating whether or 
                            not a url appeared in body)
                '''
                df0 = df.drop(columns=['sender', 'receiver', 'date', 'urls',], errors='ignore')
                
                # Fill null values in subject and body
                df0[['subject', 'body']] = df0[['subject', 'body']].fillna('<missing>')

                df1 = extract_urls.df_extractor(df0, 'body', 'url_dict')
                df2 = replace_urls.url_replacement(df1, 'body', 'url_dict', 'cleaned_body', indexed=False)
                    # indexed=False by default: if we want <url1> <url2> ..., change to True
                # Perform additional cleaning:
                    # extract url_count and distinct_url_count from url_dict
                df2['url_count'] = df2['url_dict'].apply(len)
                df2['distinct_url_count'] = df2['url_dict'].apply(lambda d: sum(d.values()))
                df2['body'] = df2['cleaned_body']
                df3 = df2.drop(columns=['cleaned_body', 'url_dict'])
                
                # # Fill null values in url counters
                # df3[['subject', 'body']] = df3[['subject', 'body']].fillna('<missing>')
                df3[['url_count', 'distinct_url_count']] = df3[['url_count', 'distinct_url_count']].fillna('<missing>')


                df_final = df3 #placeholder: df2 is wrong: fix later.

                # Append dataset to dataset list
                if level == '_train':
                    df_train_list.append(df_final)
                elif level == '_val':
                    df_val_list.append(df_final)
                else:
                    df_test_list.append(df_final)
            else:
                print(f"Dataset '{var_name}' not found.")


    # Concat all dataframes
    df_train_combined = pd.concat(df_train_list, ignore_index=True)
    df_val_combined = pd.concat(df_val_list, ignore_index=True)
    df_test_combined = pd.concat(df_test_list, ignore_index=True)


    # Save datasets
    # Make output directory if it doesn't exist
    output_dir = Path("Split_Data/Cleaned")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"cleaned_train_data.csv"
    val_path = output_dir / f"cleaned_val_data.csv"
    test_path = output_dir / f"cleaned_test_data.csv"

    df_train_combined.to_csv(train_path)
    df_val_combined.to_csv(val_path)
    df_test_combined.to_csv(test_path)

#<-- End Feature Engineering -->#


#<-- Begin DistilBERT Feature Extraction -->#

def DistilBERT_Feature_Extraction():
    import pandas as pd
    from pathlib import Path

    # Define Input Paths
    input_dir = Path("Split_Data/Cleaned")
    input_dir.mkdir(parents=True, exist_ok=True)
    training_name = "cleaned_train_data.csv"
    input_names_nt = ["cleaned_val_data.csv", "cleaned_test_data.csv"]

    # Define Output Paths
    output_dir = Path("Split_Data/Model_Ready")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_out = "train.csv"
    output_names_nt = ["val.csv", "test.csv"]

    # Define class object
    embedding_extraction = embedding_extraction()


    # Perform extraction
    # Import training data
    training_path = input_dir / training_name
    train_df_in = pd.read_csv(training_path)

    # Combine text columns, drop unecessary columns
    train_df_in['text'] = train_df_in['subject'] + " " + train_df_in['body']
    train_df_pre_pca = train_df_in.drop(columns=['body', 'subject', 'Unnamed: 0'])

    # Acquire fully cleaned training data, and pca/scaler models for the testing/validation data
    train_df_pca, pca_model, scaler_model = embedding_extraction.replace_text_with_embeddings_with_pca(train_df_pre_pca, 'text')

    # Save training csv file
    train_out_path = output_dir / train_out
    train_df_pca.to_csv(train_out_path, index=False)

    for input_name in input_names_nt:
        for output_name in output_names_nt: # This second loop is here to ensure we output to the right file
            if output_name.split('.')[0] in input_name: # Ensures that we output val to val and test to test
                # Import data
                input_path = input_dir / input_name
                df_in = pd.read_csv(input_path)

                # Combine text columns, drop unecessary columns
                df_in['text'] = df_in['subject'] + " " + df_in['body']
                df_pre_pca = df_in.drop(columns=['body', 'subject', 'Unnamed: 0'])

                # Acquire fully cleaned training/validation sets
                df_pca, _, _ = embedding_extraction.replace_text_with_embeddings_with_pca(df_pre_pca, 'text', pca=pca_model, scaler=scaler_model)

                # Save csv file
                output_path = output_dir / output_name
                df_pca.to_csv(output_path, index=False)

#<-- End DistilBERT Feature Extraction -->#

#<-- Begin Zip -->#

def To_Zip():
    import pandas as pd
    from pathlib import Path

    # Define variable names
    path = Path("Split_Data/Model_Ready")
    file_list = ['train', 'test', 'val']
    in_ext = '.csv'
    out_ext = '.csv.zip'

    # Zip csv files
    for file in file_list:
        in_file = file + in_ext
        in_path = path / in_file
        out_path = path / (file + out_ext)
        df = pd.read_csv(in_path)
        df.to_csv(out_path, index=False, compression=dict(method='zip', archive_name=in_file))

#<-- End Zip -->#



#<-- Begin Main -->#

def main():
    Data_Split()
    Feature_Engineering()
    DistilBERT_Feature_Extraction()
    To_Zip()

#<-- End Main -->#

if __name__ == "__main__":
    main()
