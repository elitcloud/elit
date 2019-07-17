# from elit.component import EnglishTokenizer
from elit.component.tokenizer import EnglishTokenizer
# from elit.component import POSFlairTagger
from elit.component import POSTagger

texts = [
    'Emory University is a private research university in Atlanta, in the U.S. state of Georgia. The university was founded as Emory College in 1836 in Oxford, Georgia, by the Methodist Episcopal Church and was named in honor of Methodist bishop John Emory.',
    'In 1915, Emory College moved to its present location in Druid Hills and was rechartered as Emory University. Emory maintained a presence in Oxford that eventually became Oxford College, a residential liberal arts college for the first two years of the Emory baccalaureate degree.[19] The university is the second-oldest private institution of higher education in Georgia and among the fifty oldest private universities in the United States.']

tok = EnglishTokenizer()
pos = POSTagger()
docs = [tok.decode(text) for text in texts]

components = [pos]

for component in components:
    docs = component.decode(docs)

print(docs)
