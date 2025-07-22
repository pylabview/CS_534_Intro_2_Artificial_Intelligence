import torch
import torch.nn as nn
word_to_ix = {}

# Class to build CBOW model for developing word embeddings.
class CBOW(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()

        # out: 1 x emdedding_dim
        # embedding_dim (int) – the size of each embedding vector, e.g., 100, for each vocab
        # Embedding(49, 100) Without Bias
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # in_features (int) – size of each input sample, e.g., 100
        # out_features (int) – size of each output sample, 128
        # Linear(in_features=100, out_features=128, bias=True)
        self.linear1 = nn.Linear(embedding_dim, 128) # Default: bias = True
        self.activation_function1 = nn.ReLU()

        # out: 1 x vocab_size
        # Linear(in_features=128, out_features=49, bias=True)
        self.linear2 = nn.Linear(128, vocab_size)
        self.activation_function2 = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view(1, -1)
        out = self.linear1(embeds)
        out = self.activation_function1(out)
        out = self.linear2(out)
        out = self.activation_function2(out)
        return out

    def get_word_emdedding(self, word):
        word = torch.tensor([word_to_ix[word]])
        return self.embeddings(word).view(1, -1)

# Let us define the vector in which the embedding will be stored.
def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

def main():
    CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
    EMDEDDING_DIM = 100

    # Here we use a small paragraph as raw text to train a word embedding.
    raw_text = """We are about to study the idea of a computational process.
    Computational processes are abstract beings that inhabit computers.
    As they evolve, processes manipulate other abstract things called data.
    The evolution of a process is directed by a pattern of rules
    called a program. People create programs to direct processes. In effect,
    we conjure the spirits of the computer with our spells.""".split()

    # First, we need to build a vocabulary!
    # There are 49 vocabularies
    vocab = set(raw_text)
    vocab_size = len(vocab)

    global word_to_ix
    # Create a dictionary, vocab is a key, and the index is the value from 0 to 48.
    word_to_ix = {word: ix for ix, word in enumerate(vocab)}
    # Create a dictionary, index is a key from 0 to 48, and the vocab is the value.
    ix_to_word = {ix: word for ix, word in enumerate(vocab)}

    # Create our dataset using a combination of context and target
    data = []
    for i in range(CONTEXT_SIZE, len(raw_text) - CONTEXT_SIZE):
        # Two words on the left and Two words on the right
        context = [raw_text[i - 2], raw_text[i - 1],
                   raw_text[i + 1], raw_text[i + 2]]
        # The word in the middle between them
        target = raw_text[i]
        # Append context and target into the data list for training.
        data.append((context, target))

    # Next, we will create the model using the CBOW class:
    model = CBOW(vocab_size, EMDEDDING_DIM)
    loss_function = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Now we are ready to do Training
    for epoch in range(50):
        total_loss = 0
        for context, target in data:
            context_vector = make_context_vector(context, word_to_ix)
            log_probs = model(context_vector)
            total_loss += loss_function(log_probs, torch.tensor([word_to_ix[target]]))

        # Optimize at the end of each epoch
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # And Then Testing
    context = ['People', 'create', 'to', 'direct']
    context_vector = make_context_vector(context, word_to_ix)
    a = model(context_vector)

    # Print result: The CBOW model uses context words ("People", "create", "to", "direct")
    # to predict the target word programs. We could get the 100-dimensional word embedding from the model
    # , where EMDEDDING_DIM = 100.
    print()
    print(f'Raw text: {" ".join(raw_text)}\n')
    print(f'Context: {context}\n')
    print(f'Prediction: {ix_to_word[torch.argmax(a[0]).item()]}')
    print()
    print(f'embedding:{model.get_word_emdedding("programs")}')


if __name__ == "__main__":
    main()