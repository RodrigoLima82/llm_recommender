# Databricks notebook source
# MAGIC %md ##Introdução
# MAGIC
# MAGIC O próximo passo na construção do nosso recomendador é aproveitar o poder de um modelo de linguagem geral de propósito amplo (LLM) para sugerir itens adicionais para um usuário. Com base no conhecimento dos itens que um cliente comprou, navegou ou manifestou interesse de alguma forma, o LLM pode acessar um reservatório de conhecimento para sugerir quais itens normalmente estão associados a esses.
# MAGIC
# MAGIC As etapas envolvidas em obter essa parte do aplicativo em funcionamento são simplesmente conectar-se a um LLM apropriado e ajustar uma solicitação que retorne os resultados corretos no formato correto.

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install databricks-genai-inference
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./00_Intro_and_Config"

# COMMAND ----------

# DBTITLE 1,Import the Required Libraries
from databricks_genai_inference import ChatCompletion

# COMMAND ----------

# MAGIC %md ##Passo 1: Conectar ao LLM
# MAGIC
# MAGIC Com a disponibilidade de uma ampla variedade de serviços proprietários e modelos de código aberto, temos várias opções para lidar com nossas necessidades de LLM. Muitos modelos frequentemente usados já foram pré-provisionados para uso dentro do Databricks como parte de nossas [APIs de modelos fundamentais](https://docs.databricks.com/en/machine-learning/foundation-models/index.html). Isso inclui o modelo [Llama2-70B-Chat da Meta](https://ai.meta.com/llama/), que está sendo amplamente adotado como um habilitador de chat robusto e de propósito geral.
# MAGIC
# MAGIC Como toda a infraestrutura já foi configurada, a conectividade a este modelo é muito simples. Basta fazer a chamada:

# COMMAND ----------

# DBTITLE 1,Text Connectivity to the LLM
response = ChatCompletion.create(
  model='llama-2-70b-chat',
  messages=[
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': 'Knock knock.'}
    ],
  max_tokens=128
  )

print(f'response.message:{response.message}')

# COMMAND ----------

# MAGIC %md ##Step 2:  Engineer the prompt
# MAGIC
# MAGIC The foundation model API greatly simplifies the process of not only connecting to a model but performing a chat task. Instead of constructing a prompt with specialized formatting, we can assemble a fairly standard [chat message payload](https://docs.databricks.com/en/machine-learning/foundation-models/api-reference.html#chat-task) to generate a response.
# MAGIC
# MAGIC As is typical in most chat applications, we will supply a system prompt and a user prompt.  The system prompt for our app might look like this:
# MAGIC
# MAGIC ```python
# MAGIC system_prompt = 'Você é um assistente de IA que funciona como um sistema de recomendação para um site de comércio eletrônico. Seja específico e limite suas respostas ao formato solicitado. Mantenha suas respostas curtas e concisas.'

# COMMAND ----------

# DBTITLE 1,Define System Prompt
system_prompt = 'Você é um assistente de IA que funciona como um sistema de recomendação para um site de comércio eletrônico. Seja específico e limite suas respostas ao formato solicitado. Mantenha suas respostas curtas e concisas.'

# COMMAND ----------

# MAGIC %md Para o prompt do usuário, precisamos incorporar uma lista de itens a partir da qual as recomendações de produtos serão geradas. Como esse prompt é dinâmico, pode ser melhor definir o prompt usando uma função:

# COMMAND ----------

# DBTITLE 1,Define Function to Build User Prompt
# define function to create prompt produce a recommended set of products
def get_user_prompt(ordered_list_of_items):

   # assemble user prompt
  prompt = None
  if len(ordered_list_of_items) > 0:
    items = ', '.join(ordered_list_of_items)
    prompt =  f"Um usuário comprou os seguintes itens: {items}. Quais seriam os próximos dez itens que ele/ela provavelmente compraria?"

  prompt += " Expresse sua resposta como um objeto JSON com uma chave 'next_items' e um valor representando sua matriz de itens recomendados."
 
  return prompt

# COMMAND ----------

# DBTITLE 1,Retrieve User Prompt
# get prompt and results
user_prompt = get_user_prompt(
    ['cachecol', 'gorro', 'protetor de ouvido', 'roupa de baixo térmica']
    )

print(user_prompt)

# COMMAND ----------

# MAGIC %md E agora podemos testar o prompt com uma chamada ao nosso LLM:

# COMMAND ----------

# DBTITLE 1,Test the Prompt
response = ChatCompletion.create(
  model='llama-2-70b-chat',
  messages=[
    {'role': 'system', 'content': system_prompt},
    {'role': 'user','content': user_prompt}
    ],
  max_tokens=128
  )
print(f'response.message:{response.message}')

# COMMAND ----------

# MAGIC %md Obter a resposta do modelo com listas na estrutura desejada e com conteúdo relevante é complicado. (Por exemplo, não conseguimos obter uma estrutura de dicionário válida ao solicitar um dicionário Python, mas descobrimos que solicitar um objeto JSON resolveu o problema.) Você precisará experimentar uma variedade de palavras e frases para obter os resultados desejados.

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | Llama2-70B-Chat | A pretrained and fine-tuned generative text model with 70 billion parameters |  Meta |https://ai.meta.com/resources/models-and-libraries/llama-downloads                       |
