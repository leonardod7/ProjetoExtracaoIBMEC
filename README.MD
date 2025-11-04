# 📊 Base de Dados – Olist Brazilian E-Commerce

Esta base de dados é proveniente do **Brazilian E-Commerce Public Dataset by Olist**, amplamente utilizado em estudos acadêmicos e projetos de ciência de dados. Ela contém informações reais de transações de comércio eletrônico realizadas entre **2016 e 2018** no Brasil, cobrindo o ciclo completo de vendas online:

- **Pedidos (Orders):** detalhes de cada compra realizada na plataforma.  
- **Itens do pedido (Order Items):** produtos vendidos, preços e frete.  
- **Clientes (Customers):** informações geográficas e identificadores de clientes.  
- **Sellers (Vendedores):** localização e características dos lojistas que venderam na plataforma.  
- **Produtos (Products):** descrição e categoria dos itens ofertados.  
- **Pagamentos (Payments):** formas e valores de pagamento utilizados.  
- **Avaliações (Reviews):** notas e comentários dos clientes sobre os pedidos.  
- **Geolocalização (Geolocation):** dados de latitude/longitude para CEPs de clientes e sellers.  

Após o processo de integração e tratamento, os dados foram organizados em uma **tabela fato denormalizada** (`olist_order_lines_denorm.csv`) e **dimensões auxiliares** (`dim_customers.csv`, `dim_products.csv`, `dim_sellers.csv`), facilitando análises descritivas e a construção de dashboards.

---

## 📈 Destaques da Base

- **Período coberto:** Setembro/2016 a Setembro/2018  
- **Pedidos únicos:** ~98 mil  
- **Itens vendidos:** ~112 mil  
- **Categorias de produto:** 70+ categorias (ex.: cama_mesa_banho, beleza_saude, esporte_lazer)  
- **Clientes e Sellers:** abrangem praticamente todos os estados do Brasil, com concentração em São Paulo (SP).

---

## 🧠 Objetivo do Projeto

O objetivo deste projeto é desenvolver um **pipeline de ETL (Extract, Transform, Load)** para integrar, tratar e consolidar os dados do Olist, gerando uma **tabela fato denormalizada** e **dimensões auxiliares**.  
O resultado final serve de base para análises descritivas, preditivas e para a criação de dashboards no **Power BI** e **Google Looker Studio**.

---

## 🧩 Pipeline de ETL

O script `build_olist_model.py` realiza automaticamente as seguintes etapas:

1. **Extração:**  
   Carrega todas as tabelas do dataset (`orders`, `order_items`, `payments`, `reviews`, `customers`, `sellers`, `products`, `geolocation`).

2. **Transformação:**  
   - Converte tipos de dados e padroniza colunas.  
   - Calcula o **atraso na entrega (`delay_days`)** com base na diferença entre a data real e a data estimada.  
   - Agrega dados de pagamento e avaliação por pedido.  
   - Integra informações geográficas por CEP.  
   - Gera métricas de valor total (`item_gmv = price + freight_value`).

3. **Carga (Load):**  
   Exporta os resultados em:
   - `olist_order_lines_denorm.csv` (tabela fato)  
   - `dim_customers.csv`, `dim_sellers.csv`, `dim_products.csv`  
   - `metrics_model_comparison.csv` (comparação entre baseline e regressão linear)

---

## 📈 Resultados

- **Total de linhas:** ~112.000  
- **Atributos:** 42 colunas integradas  
- **Atraso médio:** 3 dias  
- **Entregas adiantadas:** 27% dos pedidos  
- **Correlação atraso × nota:** -0,22  
- **Nota média:**  
  - Entregas pontuais/adiantadas: 4,3  
  - Entregas atrasadas: 3,6  
- **Estados com maior volume:** SP, MG e RJ (60% dos pedidos)

---

## 🔍 Comparação de Modelos

| Modelo | Descrição | MAE (Erro Médio Absoluto) |
|:-------|:-----------|:--------------------------|
| **Baseline** | Média global das avaliações | 0.67 |
| **Regressão Linear** | `review_score ~ delay_days` | 0.61 |

O modelo linear apresentou melhor desempenho e confirmou a **relação negativa entre atraso e satisfação do cliente**.

---

## 👥 Integrantes do Grupo
- Emilly Fernandes
- Leonardo dos Santos
- Tadeu Tovar
