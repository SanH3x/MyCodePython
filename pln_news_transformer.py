import requests
from bs4 import BeautifulSoup
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer  # Frequência (LSA)
from sumy.summarizers.luhn import LuhnSummarizer # Luhn
from transformers import pipeline
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.corpus import mac_morpho
from nltk.tokenize import word_tokenize
import string


def extract_text_from_url(url):
    # Realiza uma requisição HTTP para as URL fornecidas
    response = requests.get(url)

    # Analisa o conteúdo HTML da página
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extração de todo o texto da página
    paragraphs = soup.find_all('p')

    # Concatena os textos de todos os parágrafos extraídos em uma única string
    text = ' '.join([para.get_text() for para in paragraphs])

    # Retorna os textos extraído
    return text


def summarize_by_frequency(text, sentence_count=5):
    # Converte o texto em um formato processável pelo sumarizador
    parser = PlaintextParser.from_string(text, Tokenizer('english'))

    # Inicializa o sumarizador LSA (Análise Latente de Semântica)
    summarizer = LsaSummarizer()

    # Gera o resumo com base na quantidade de sentenças especificada
    summary = summarizer(parser.document, sentence_count)

    # Concatena as sentenças resumidas em uma única string
    return ' '.join([str(sentence) for sentence in summary])


def summarize_by_luhn(text, sentence_count=5):
    # Converte o texto em um formato processável pelo sumarizador
    parser = PlaintextParser.from_string(text, Tokenizer('english'))

    # Inicializa o sumarizador Luhn
    summarizer = LuhnSummarizer()

     # Gera o resumo com base na quantidade de sentenças especificada
    summary = summarizer(parser.document, sentence_count)

    # Concatena as sentenças resumidas em uma única string
    return ' '.join([str(sentence) for sentence in summary])


def summarize_by_deep_model(text, max_length=150, min_length=40):
    # Inicializa o pipeline de sumarização da biblioteca transformers
    summarizer = pipeline("summarization")

    # Trunca o texto de entrada se ele exceder o comprimento máximo do modelo
    tokenizer = summarizer.tokenizer

    # Supondo que o modelo tenha comprimento máximo de 512 tokens. Isso pode ser diferente para outros modelos.
    max_input_length = 512

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_length)

    summary_ids = summarizer.model.generate(inputs["input_ids"], max_length=max_length, min_length=min_length)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Retorna o texto resumido
    return summary


# Baixar stopwords se necessário
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('mac_morpho')

# Carregar modelo do spaCy para POS tagging
nlp = spacy.load('en_core_web_sm')


def process_text(text):
    # Tokenização dos textos
    tokens = word_tokenize(text)

    # Remoção das stopwords e pontuação
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words and word not in string.punctuation]

    # Etiquetagem de POS
    doc = nlp(' '.join(tokens))
    pos_tags = [(token.text, token.pos_) for token in doc]

    return pos_tags


def preprocess_text(text):
    # Tokenização dos textos em palavras
    tokens = nltk.word_tokenize(text)

    # Remoção das stopwords e pontuação
    stop_words = set(stopwords.words('english'))
    tokens_clean = [word for word in tokens if word.isalnum() and word.lower() not in stop_words]

    return tokens_clean


# Função para aplicar etiquetadores (Default, Unigram, Bigram, Trigram) no texto tokenizado
def apply_taggers(text):
    # Tokenização dos textos
    tokens = nltk.word_tokenize(text)

    # Aplica o etiquetador padrão
    etiqPadrao = nltk.tag.DefaultTagger('')
    default_tags = etiqPadrao.tag(tokens)

    # Preparação de dados: 90% treino, 10% teste usando o corpus mac_morpho
    prop = int(0.1 * len(mac_morpho.tagged_sents()))
    treino = mac_morpho.tagged_sents()[prop:]
    teste = mac_morpho.tagged_sents()[:prop]

    # Etiquetador padrão
    etiq1 = nltk.DefaultTagger('')
    print('Basic Tagger:', etiq1.accuracy(teste))

    # Etiquetador Unigram
    etiq2 = nltk.UnigramTagger(treino, backoff=etiq1)
    print('UNIGRAM Tagger:', etiq2.accuracy(teste))

    # Etiquetador Bigram
    etiq3 = nltk.BigramTagger(treino, backoff=etiq2)
    print('BIGRAM Tagger:', etiq3.accuracy(teste))

    # Etiquetador Trigram
    etiq4 = nltk.TrigramTagger(treino, backoff=etiq3)
    print('TRIGRAM Tagger:', etiq4.accuracy(teste))

    # Etiqueta o texto fornecido com o etiquetador trigram (mais preciso)
    trigram_tags = etiq4.tag(tokens)

    # Retorna as etiquetas padrões e as etiquetas com trigram
    return default_tags, trigram_tags


# Função para gerar um relatório HTML com os resultados de ambas as URLs
def generate_html_report(original_texts, freq_summaries, luhn_summaries, deep_summaries, pos_tags_list, tagger_results):
    html_content = """
    <html>
    <head><title>PLN Report - Comparação de URLs</title></head>
    <body>
        <h1>Relatório de Processamento de Linguagem Natural</h1>
    """

    # Iterar pelos textos das duas URLs e adicionar ao HTML
    for i in range(2):
        html_content += f"""
        <h2>Texto Original da URL {i+1}</h2>
        <p>{original_texts[i]}</p>

        <h3>Sumarização por Frequência (LSA) - URL {i+1}</h3>
        <p>{freq_summaries[i]}</p>

        <h3>Sumarização por Luhn - URL {i+1}</h3>
        <p>{luhn_summaries[i]}</p>

        <h3>Sumarização Profunda - URL {i+1}</h3>
        <p>{deep_summaries[i]}</p>

        <h3>Etiquetagem POS (Part of Speech) - URL {i+1}</h3>
        <ul>
            {''.join([f'<li>{word} - {pos}</li>' for word, pos in pos_tags_list[i]])}
        </ul>

        <h3>Etiquetadores NLTK - URL {i+1}</h3>
        <p>Etiquetador Padrão: {tagger_results[i][0]}</p>
        <p>Etiquetador Trigram: {tagger_results[i][1]}</p>
        <hr>
        """

    html_content += """
    </body>
    </html>
    """

    with open('pln_report2.html', 'w') as file:
        file.write(html_content)
    print("Relatório gerado: pln_report2.html")


# Função principal para processar duas URLs
def main(url1, url2):
    # Etapa 1: Extração do texto das duas URLs
    text1 = extract_text_from_url(url1)
    text2 = extract_text_from_url(url2)

    # Armazenar os textos originais
    original_texts = [text1, text2]

    # Etapa 2: Sumarizações
    freq_summaries = [summarize_by_frequency(text1), summarize_by_frequency(text2)]
    luhn_summaries = [summarize_by_luhn(text1), summarize_by_luhn(text2)]
    deep_summaries = [summarize_by_deep_model(text1), summarize_by_deep_model(text2)]

    # Etapa 3: Processamento de Linguagem Natural
    pos_tags_list = [process_text(text1), process_text(text2)]

    # Etapa 4: Aplicar os etiquetadores NLTK (Default, Unigram, Bigram, Trigram)
    tagger_results = ['Notícia 1', apply_taggers(text1), 'Notícia 2', apply_taggers(text2)]

    # Etapa 5: Geração do relatório HTML
    generate_html_report(original_texts, freq_summaries, luhn_summaries, deep_summaries, pos_tags_list, tagger_results)


# URLs de exemplo
url1 = ''
url2 = ''

# Chama a função principal
main(url1, url2)