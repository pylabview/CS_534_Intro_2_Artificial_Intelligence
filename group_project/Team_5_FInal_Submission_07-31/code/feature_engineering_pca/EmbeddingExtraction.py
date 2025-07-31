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