# Databricks notebook source
# MAGIC %md O objetivo deste caderno é introduzir o acelerador de solução do Recomendador LLM e fornecer configurações para os vários cadernos que o compõem. Este caderno foi desenvolvido usando um cluster Databricks ML 14.2.

# COMMAND ----------

# MAGIC %md ##Introdução
# MAGIC
# MAGIC Os recomendadores empregam várias estratégias para sugerir produtos que podem atrair um cliente. Com este recomendador, estamos considerando um conjunto de produtos que um cliente já comprou ou mostrou preferência e estamos usando nosso conhecimento desses produtos para sugerir itens adicionais e relevantes.
# MAGIC
# MAGIC A associação entre esses dois conjuntos de produtos, *ou seja*, aqueles que são preferidos e aqueles que são sugeridos, é baseada em um entendimento geral de como os itens se relacionam dentro de uma determinada cultura. Por exemplo, para muitos consumidores, se entendêssemos que estavam comprando um par de *luvas*, um *cachecol* e um *casaco*, isso poderia sugerir que *botas de inverno*, um *chapéu isolante* ou *protetores de ouvido* poderiam ser de interesse, pois esses itens se relacionam conceitualmente com a proteção contra o frio.
# MAGIC
# MAGIC Esse tipo de recomendador não leva em conta as relações entre os itens comprados como poderia ser observado nos dados históricos de uma organização, de modo que o famoso (embora espúrio) padrão de [*fraldas e cerveja*](https://tdwi.org/articles/2016/11/15/beer-and-diapers-impossible-correlation.aspx) talvez nunca surja. Ainda assim, este recomendador reflete como um indivíduo razoável poderia sugerir itens a outro e pode proporcionar uma experiência útil e agradável em alguns contextos.

# COMMAND ----------

# MAGIC %md A outra limitação desta abordagem é que os itens sugeridos são altamente generalizados, *por exemplo*, *botas de inverno* é uma categoria ampla de produtos e não um SKU específico. Além disso, o LLM não tem conhecimento se um item sugerido está disponível para compra em um determinado ponto de venda.
# MAGIC
# MAGIC Para superar essa limitação, precisamos cruzar os itens sugeridos generalizados com os itens específicos em nosso inventário de produtos. Uma recomendação geral de *botas de inverno* pode se alinhar bem com vários *botas de neve*, *botas isoladas* ou até *galochas impermeáveis* encontradas em nosso catálogo de produtos, dependendo da nossa tolerância em relação à similaridade dos itens.
# MAGIC
# MAGIC Para apoiar isso, podemos pegar as informações descritivas associadas a cada um dos produtos em nosso inventário e convertê-las (usando um LLM) em um embedding. O embedding captura um mapeamento de cada item para os vários *conceitos* aprendidos pelo LLM à medida que foi exposto aos dados de treinamento. Podemos então converter uma sugestão generalizada como *botas de inverno* em um embedding usando o mesmo modelo e calcular a similaridade entre o embedding do produto sugerido e os embeddings de vários itens em nosso inventário de produtos para identificar aqueles itens mais *conceitualmente* relacionados àquela sugestão.
# MAGIC `

# COMMAND ----------

# MAGIC %md Com esses conceitos em mente, abordaremos este acelerador de solução conectando-se a um LLM e desenvolvendo um prompt para acionar sugestões generalizadas de produtos. Em seguida, converteremos os detalhes do nosso produto em um banco de dados pesquisável de embeddings. E, finalmente, uniremos essas duas partes da solução para permitir a implantação de um recomendador robusto em uma infraestrutura de aplicativos.

# COMMAND ----------

# MAGIC %md ## Configuration
# MAGIC
# MAGIC The following settings are used across the various notebooks found in this solution accelerator:

# COMMAND ----------

# DBTITLE 1,Instantiate Config
if 'config' not in locals().keys():
  config = {}

# COMMAND ----------

# DBTITLE 1,Product Database
# set catalog
config['catalog'] = 'rodrigo_catalog'
_ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {config['catalog']}")
_ = spark.sql(f"USE CATALOG {config['catalog']}")

# set schema
config['schema'] = 'llm_recommender'
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {config['schema']}")
_ = spark.sql(f"USE SCHEMA {config['schema']}")

# COMMAND ----------

# DBTITLE 1,Vector Search Index
config['vs index'] = 'product_index'

# COMMAND ----------

# DBTITLE 1,Model
config['embedding_model_name'] = 'llm_recommender_embeddings'

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | Llama2-70B-Chat | A pretrained and fine-tuned generative text model with 70 billion parameters |  Meta |https://ai.meta.com/resources/models-and-libraries/llama-downloads                       |
