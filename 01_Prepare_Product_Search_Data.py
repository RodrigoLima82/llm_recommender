# Databricks notebook source
# MAGIC %md ##Introdução
# MAGIC
# MAGIC Neste notebook, usaremos informações descritivas sobre os produtos que pretendemos apresentar ao usuário para criar um conjunto pesquisável de embeddings. Esses embeddings serão usados para permitir uma pesquisa rápida e flexível de nossos produtos.
# MAGIC
# MAGIC Para realizar esse trabalho, devemos carregar os dados sobre nossos produtos em uma tabela de banco de dados. Em seguida, devemos configurar um modelo com o qual converteremos informações descritivas sobre esses produtos em embeddings. Em seguida, acionaremos um fluxo de trabalho contínuo que manterá nossos embeddings pesquisáveis, ou seja, nosso índice de pesquisa de vetores, em sincronia com a tabela.

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./00_Intro_and_Config"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import datasets # hugging face datasets
datasets.utils.logging.disable_progress_bar() # disable progress bars from hugging face

from sentence_transformers import SentenceTransformer

from databricks.vector_search.client import VectorSearchClient

import mlflow

import pyspark.sql.functions as fn

import pandas as pd
import json

import requests

import time

# COMMAND ----------

# MAGIC %md ##Passo 1: Carregar o Conjunto de Dados
# MAGIC
# MAGIC O conjunto de dados que iremos utilizar é o [Red Dot Design Award dataset](https://huggingface.co/datasets/xiyuez/red-dot-design-award-product-description), disponível através do HuggingFace. Este conjunto de dados contém informações sobre produtos premiados, incluindo texto descritivo que podemos usar para pesquisas. Vamos tratá-lo como se fosse nosso conjunto real de produtos disponíveis para venda aos clientes:

# COMMAND ----------

# DBTITLE 1,Import the HuggingFace Dataset
# import dataset to hugging face dataset
ds = datasets.load_dataset("xiyuez/red-dot-design-award-product-description") 

print(ds)

# COMMAND ----------

# MAGIC %md O HuggingFace disponibiliza esse conjunto de dados como um dicionário de conjunto de dados. Vamos persisti-lo como uma tabela Delta Lake, pois essa é a forma mais comum de os usuários do Databricks acessarem informações de produtos dentro do lakehouse.
# MAGIC
# MAGIC Por favor, observe que estamos definindo a tabela de destino para esses dados antecipadamente para que possamos adicionar um [campo de identidade](https://www.databricks.com/blog/2022/08/08/identity-columns-to-generate-surrogate-keys-are-now-available-in-a-lakehouse-near-you.html) a ela. Criar um campo de identificação dessa maneira simplifica a criação de identificadores exclusivos para cada item em nosso conjunto de dados:

# COMMAND ----------

# DBTITLE 1,Persist as Delta Lake Table
# drop any pre-existing indexes on table
vs_client = VectorSearchClient()
try:
  vs_client.delete_index(f"{config['catalog']}.{config['schema']}.{config['vs index']}")
except:
  print('Ignoring error message associated with vs index deletion ...')
  pass


# create table to hold product info
_ = spark.sql('''
  CREATE OR REPLACE TABLE products (
    id bigint GENERATED ALWAYS AS IDENTITY,
    product string,
    category string,
    description string,
    text string
    )'''
  )

# add product info to table
_  = (
  spark
    .createDataFrame(ds['train']) 
    .select('product','category','description','text') # fields in correct order 
    .write
    .format('delta')
    .mode('append')
    .saveAsTable('products')
  )

# read data from table
products = spark.table('products')

# display table contents
display( products )

# COMMAND ----------

# MAGIC %md É importante observar que o campo de texto contém os nomes e descrições concatenados de nossos produtos. Este é o campo com base no qual faremos nossas pesquisas posteriores.

# COMMAND ----------

# MAGIC %md ##Passo 2: Preencher o Armazenamento de Vetores
# MAGIC
# MAGIC Para habilitar nosso aplicativo, precisamos converter as informações de pesquisa de produtos em embeddings armazenados em um armazenamento de vetores pesquisável. Para isso, faremos uso do armazenamento de vetores integrado do Databricks. Mas para criar embeddings compreendidos pelo armazenamento de vetores, precisamos implantar um modelo de embedding em um ponto de extremidade de serviço de modelo do Databricks.

# COMMAND ----------

# MAGIC %md ###Passo 2a: Implantar Modelo no Registro do MLFlow
# MAGIC
# MAGIC Para permitir uma busca eficiente das informações de nossos produtos, precisamos converter o texto descritivo associado a cada registro em um embedding. Cada embedding é uma representação numérica do conteúdo dentro do texto. Podemos usar qualquer modelo de transformer capaz de converter texto em um embedding para esta etapa. Estamos usando o [modelo all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) porque esse modelo produz um embedding (vetor) razoavelmente compacto e foi treinado para cenários de linguagem de propósito geral:

# COMMAND ----------

# DBTITLE 1,Download HuggingFace Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# COMMAND ----------

# MAGIC %md Para entender melhor o que esse modelo produz, podemos pedir a ele para codificar algumas strings simples. Observe como cada uma é convertida em uma matriz de ponto flutuante representando um embedding. Essas matrizes fornecem um mapa do conteúdo dentro de cada unidade de texto fornecida com base nas informações codificadas no modelo baixado:

# COMMAND ----------

# DBTITLE 1,Generate a Sample Embedding
# some sample sentences to convert to embeddings
sentences = [
  "This is an example sentence", 
  "Each sentence is converted into an embedding",
  "An embedding is nothing more than a large, numerical representation of the contents of a unit of text"
  ]

# convert the sentences to embeddings
embeddings = model.encode(sentences)

# display the embeddings
display(embeddings)

# COMMAND ----------

# MAGIC %md Usando as frases de exemplo e os embeddings resultantes, podemos gerar uma assinatura para nosso modelo. Uma assinatura é apenas um esquema leve que define a estrutura esperada das entradas e saídas para um determinado modelo. Essa assinatura nos ajudará a implantar nosso modelo em um endpoint de serviço de modelo posteriormente neste notebook:

# COMMAND ----------

# DBTITLE 1,Generate a Model Signature
signature = mlflow.models.signature.infer_signature(sentences, embeddings)
print(signature)

# COMMAND ----------

# MAGIC %md Agora podemos registrar o modelo juntamente com sua assinatura no [registro do MLFlow](https://docs.databricks.com/en/mlflow/model-registry.html). Este é um passo fundamental na publicação do modelo em um ponto de extremidade de serviço de modelo. Observe que estamos registrando este modelo usando o [*sentence_transformers* modelo flavor](https://mlflow.org/docs/latest/python_api/mlflow.sentence_transformers.html) dentro do MLFlow, que é especificamente projetado para trabalhar com essa classe de modelo:

# COMMAND ----------

# DBTITLE 1,Register Model with MLFlow
# identify the experiment that will house this mlflow run
user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment(f"/Users/{user_name}/{config['embedding_model_name']}")
             
# initiate mlflow run to log model to experiment
with mlflow.start_run(run_name=config['embedding_model_name']) as run:

  model_info = mlflow.sentence_transformers.log_model(
    model,
    artifact_path='model',
    signature=signature,
    input_example=sentences,
    registered_model_name=config['embedding_model_name']
    )

# COMMAND ----------

# MAGIC %md Com nosso modelo registrado no MLFlow, agora podemos determinar a versão do modelo associada a essa implantação. À medida que implantamos versões subsequentes deste modelo, o número da versão do MLFlow será incrementado:

# COMMAND ----------

# DBTITLE 1,Get Registered Model to Version
# connect to mlflow
mlf_client = mlflow.MlflowClient()

# search for the model versions with the specified name
model_versions = mlf_client.search_model_versions(f"name = 'rodrigo_catalog.llm_recommender.{config['embedding_model_name']}'")

# sort the model versions by version number in descending order
sorted_versions = sorted(model_versions, key=lambda x: x.version, reverse=True)

# get the latest version
latest_version = sorted_versions[0]

# get the version number
model_version = latest_version.version

print(model_version)

# COMMAND ----------

# MAGIC %md ###Passo 2b: Criar Endpoint de Serviço do Modelo
# MAGIC
# MAGIC Com nosso modelo implantado no registro do MLFlow, agora podemos enviar o modelo para um endpoint de serviço do modelo do Databricks. Esse endpoint permitirá a população do vetor de armazenamento posteriormente neste notebook:

# COMMAND ----------

# DBTITLE 1,Configuration Values for Model Serving Endpoint
#name used to reference the model serving endpoint
endpoint_name = config['embedding_model_name']

# get url of this workspace where the model serving endpoint will be deployed
workspace_url = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

# MAGIC %md Para implantar nosso ponto de extremidade de serviço de modelo, podemos usar a interface do usuário de serviço de modelo ou a API REST do Databricks (administrativa), conforme documentado [aqui](https://docs.databricks.com/en/machine-learning/model-serving/create-manage-serving-endpoints.html). Optamos por usar a API REST, pois ela simplifica o processo de implantação. No entanto, requer o uso de um token de acesso pessoal ou um principal de serviço para autenticar na API. Optamos por usar um token de acesso pessoal por uma questão de simplicidade, mas recomendamos o uso de principais de serviço em todas as implantações de produção. Mais informações sobre as opções de autenticação estão disponíveis [aqui](https://docs.databricks.com/en/dev-tools/auth.html).
# MAGIC
# MAGIC O token de acesso pessoal que configuramos está protegido como um segredo do Databricks com um escopo de *llm_recommender* e uma chave de *embedding_model_endpoint_pat*. Mais detalhes sobre a configuração de um segredo podem ser encontrados [aqui](https://docs.databricks.com/en/security/secrets/index.html), mas os comandos básicos do Databricks CLI para configurar isso são os seguintes:

# COMMAND ----------

# DBTITLE 1,Authentication for Databricks REST API
# personal access token used by model serving endpoint to retrieve the model from mlflow registry
token = dbutils.secrets.get(scope="llm_recommender", key="embedding_model_endpoint_pat")

# COMMAND ----------

```markdown
%md Com essas informações, agora podemos fazer as chamadas necessárias para implantar um ponto de extremidade de serviço de modelo dentro do Databricks. Usando o endpoint [serving-endpoint](https://docs.databricks.com/en/machine-learning/model-serving/create-manage-serving-endpoints.html#api-workflow), primeiro verificaremos se o ponto de extremidade já foi implantado. Se não tiver sido, implantaremos o ponto de extremidade e esperaremos até que ele entre em um estado pronto:

**NOTA** No código abaixo, estamos configurando o ponto de extremidade para permanecer ativo, *ou seja*, NÃO escalar para zero. Isso incorrerá em cobranças contínuas. Certifique-se de usar o código no final deste caderno para descartar o ponto de extremidade assim que terminar de trabalhar neste acelerador de solução.

# COMMAND ----------

# DBTITLE 1,Deploy Model Serving Endpoint
# header for databricks rest api calls
headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}

# databricks rest api endpoint for deployment
base_url = f'{workspace_url}/api/2.0/serving-endpoints'

# does the model serving endpoint already exist?
results = requests.request(method='GET', headers=headers, url=f"{base_url}/{endpoint_name}")

# if endpoint exists ...
if results.status_code==200:
  print('endpoint already exists')

# otherwise, create an endpoint
else:
  print('creating endpoint')

  # configuration for model serving endpoint
  endpoint_config = {
    "name": endpoint_name,
    "config": {
      "served_models": [{
        "name": f"{config['embedding_model_name'].replace('.', '_')}_{1}",
        "model_name": config['embedding_model_name'],
        "model_version": model_version,
        "workload_type": "CPU",
        "workload_size": "Small",
        "scale_to_zero_enabled": True, # you may want to set this to false to minimize startup times
      }]
    }
  }

  # convert dictionary to json
  endpoint_json = json.dumps(endpoint_config, indent='  ')

  # send json payload to databricks rest api
  deploy_response = requests.request(method='POST', headers=headers, url=base_url, data=endpoint_json)

  # get response from databricks api
  if deploy_response.status_code != 200:
    raise Exception(f'Request failed with status {deploy_response.status_code}, {deploy_response.text}')


# wait for endpoint to get into responsive state
timeout_seconds = 30 * 60 # minutes * seconds/minute 
stop_time = time.time() + timeout_seconds
waiting = True

while time.time() <= stop_time:

  # check on status of endpoint
  query_response = requests.request(method='GET', headers=headers, url=f"{base_url}/{endpoint_name}")

  status = query_response.json()['state']['ready']
  print(f"endpoint status: {status}")

    # if status is not ready, then sleep and try again
  if status=='NOT_READY':
    time.sleep(30)
  else: # otherwise stop looping
    waiting = False
    break

if waiting:
  raise Exception(f'Timeout expired waiting for endpoint to achieve a ready state.  Consider elevating the timeout setting.')

# COMMAND ----------

# MAGIC %md Com o ponto de extremidade persistido, agora podemos verificar se ele está produzindo embeddings para nós:
# MAGIC
# MAGIC **NOTA** Se você já configurou o ponto de extremidade anteriormente e o configurou com *scale_to_zero_enabled* definido como True, o ponto de extremidade pode estar adormecido quando você solicitar uma resposta abaixo. Definir um tempo limite apropriado permitirá que o ponto de extremidade acorde e responda.

# COMMAND ----------

# DBTITLE 1,Assemble Testing Payload for Endpoint
# assemble a few test sentences
sentences = ['This is a test', 'This is only a test']

# assemble them as a payload as expected by the model serving endpoint
ds_dict = {'dataframe_split': pd.DataFrame(pd.Series(sentences)).to_dict(orient='split')}
data_json = json.dumps(ds_dict, allow_nan=True)

print(data_json)

# COMMAND ----------

# DBTITLE 1,Test the Endpoint
invoke_headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
invoke_url = f'{workspace_url}/serving-endpoints/{endpoint_name}/invocations'

# test the model serving endpoint with the assembled testing data
invoke_response = requests.request(method='POST', headers=invoke_headers, url=invoke_url, data=data_json, timeout=300)

# display results from endpoint
if invoke_response.status_code != 200:
  raise Exception(f'Request failed with status {invoke_response.status_code}, {invoke_response.text}')

print(invoke_response.text)

# COMMAND ----------

# MAGIC %md ###Passo 2c: Preencher o Armazenamento de Vetores
# MAGIC
# MAGIC Com nosso modelo de incorporação implantado em um ponto de extremidade de serviço de modelo, agora podemos definir um fluxo de trabalho para converter dados em nossa tabela de produtos em entradas em nosso armazenamento de vetores. A criação e manutenção do índice ocorrerão como parte de uma automação contínua. Ele detectará alterações em nossa tabela de produtos lendo o registro de alterações associado a ela. Para garantir que o [registro de alterações](https://docs.databricks.com/en/delta/delta-change-data-feed.html) esteja habilitado, podemos alterar a definição da tabela da seguinte forma:
# MAGIC
# MAGIC **NOTA** A detecção de alterações só funciona com tabelas persistidas no formato Delta Lake.

# COMMAND ----------

# DBTITLE 1,Enable the Change Log on our Product Table
_ = spark.sql("ALTER TABLE products SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# COMMAND ----------

# MAGIC %md Agora podemos configurar um trabalho para converter dados em incorporações continuamente. Isso é feito criando um ponto de extremidade referenciável para o armazenamento de vetores e um índice associado a ele:

# COMMAND ----------

# DBTITLE 1,Instantiate Vector Search Client
vs_client = VectorSearchClient()

# COMMAND ----------

# DBTITLE 1,Create Vector Search Endpoint
#name used for vector search endpoint
endpoint_name = "one-env-shared-endpoint-2"

# check if exists
endpoint_exists = True
try:
    vs_client.get_endpoint(endpoint_name)
except:
    pass
    endpoint_exists = False

# create vs endpoint
if not endpoint_exists:
    vs_client.create_endpoint(
        name=endpoint_name,
        endpoint_type="STANDARD" # or PERFORMANCE_OPTIMIZED, STORAGE_OPTIMIZED
    )

# COMMAND ----------

# DBTITLE 1,Create Vector Search Index
# check if index exists
index_exists = False
try:
  vs_client.get_index(index_name=f"{config['catalog']}.{config['schema']}.{config['vs index']}", endpoint_name=endpoint_name)
  index_exists = True
except:
  print('Ignoring error message ...')
  pass


if not index_exists:
  # connect delta lake table to vector store index table
  vs_client.create_delta_sync_index(
    endpoint_name=endpoint_name,
    source_table_name=f"{config['catalog']}.{config['schema']}.products",
    primary_key="id", # primary identifier in source table
    embedding_source_column="text", # field to index in source table
    index_name=f"{config['catalog']}.{config['schema']}.{config['vs index']}",
    pipeline_type='TRIGGERED',
    embedding_model_endpoint_name = config['embedding_model_name'] # model serving endpoint to use to create the embeddings
    )

# COMMAND ----------

# MAGIC %md A indexação funciona como parte de um trabalho em segundo plano. Levará algum tempo para que o trabalho seja iniciado e comece a gerar embeddings. Enquanto aguardamos isso, o trabalho estará em estado de *provisionamento*. Precisaremos aguardar a conclusão antes de prosseguir com o restante deste notebook:

# COMMAND ----------

# DBTITLE 1,Wait for Vector Store Index to Start Processing Data
timeout_seconds = 120 * 60  # minutes * seconds/minute
stop_time = time.time() + timeout_seconds
waiting = True

# get index
idx = vs_client.get_index(index_name=f"{config['catalog']}.{config['schema']}.{config['vs index']}", endpoint_name=endpoint_name)

# wait for index to complex indexing
while time.time() <= stop_time:

  # get state of index
  is_ready = idx.describe()['status']['ready']

  # if not ready, wait ...
  if is_ready:
    print('Ready')
    waiting = False
    break
  else:
    print('Waiting...')
    time.sleep(60)
   
# if exited loop because of time out, raise error
if waiting:
  raise Exception(f'Timeout expired waiting for index to be provisioned.  Consider elevating the timeout setting.')

# COMMAND ----------

# MAGIC %md ##Passo 3: Pesquisar no Vector Store
# MAGIC
# MAGIC Com o vector store populado, podemos realizar uma pesquisa simples da seguinte forma:

# COMMAND ----------

# DBTITLE 1,Locate Relevant Content in Vector Store Index
# connect to index
idx = vs_client.get_index(index_name=f"{config['catalog']}.{config['schema']}.{config['vs index']}", endpoint_name=endpoint_name)

# search the vector store for related items
idx.similarity_search(
  query_text = "winter boots",
  columns = ["id", "text"], # columns to return
  num_results = 5
  )

# COMMAND ----------

# MAGIC %md Você notará que os resultados básicos da pesquisa incluem muitos metadados. Se você deseja obter apenas os itens recuperados, essas informações são encontradas nas chaves `results` e `data array`:

# COMMAND ----------

# DBTITLE 1,Get Just the Results
search_results = idx.similarity_search(
  query_text = "winter boots",
  columns = ["id", "text"], # columns to return
  num_results = 5
  )

print(search_results['result']['data_array'])

# COMMAND ----------

# MAGIC %md Observe que a pesquisa por vetor não corresponde necessariamente com base em correspondências de palavras, mas sim mapeia o texto de consulta fornecido em uma incorporação que representa como o item fornecido se relaciona com os conceitos que ele aprendeu ao processar grandes volumes de texto. Como resultado, é possível que a pesquisa encontre itens *relacionados* que não correspondam exatamente ao termo de pesquisa, mas que tenham uma associação mais geral. Para esse tipo de recomendador, esse tipo de pesquisa expansiva é aceitável, pois estamos tentando não corresponder a itens exatos, mas sim aproveitar essas associações soltas para expandir o conjunto de produtos relevantes que podemos apresentar a um cliente.

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | Llama2-70B-Chat | A pretrained and fine-tuned generative text model with 70 billion parameters |  Meta |https://ai.meta.com/resources/models-and-libraries/llama-downloads                       |
