📊 Base de Dados – Olist Brazilian E-Commerce

Esta base de dados é proveniente do Brazilian E-Commerce Public Dataset by Olist, amplamente utilizado em estudos acadêmicos e projetos de ciência de dados. Ela contém informações reais de transações de comércio eletrônico realizadas entre 2016 e 2018 no Brasil, cobrindo o ciclo completo de vendas online:

- > Pedidos (Orders): detalhes de cada compra realizada na plataforma.
- > Itens do pedido (Order Items): produtos vendidos, preços e frete.
- > Clientes (Customers): informações geográficas e identificadores de clientes.
- > Sellers (Vendedores): localização e características dos lojistas que venderam na plataforma.
- > Produtos (Products): descrição e categoria dos itens ofertados.
- > Pagamentos (Payments): formas e valores de pagamento utilizados.
- > Avaliações (Reviews): notas e comentários dos clientes sobre os pedidos.
- > Geolocalização (Geolocation): dados de latitude/longitude para CEPs de clientes e sellers.

Após o processo de integração e tratamento, os dados foram organizados em uma tabela fato denormalizada (olist_order_lines_denorm.csv) e dimensões auxiliares (dim_customers.csv, dim_products.csv, dim_sellers.csv), facilitando análises descritivas e a construção de dashboards.

📈 Destaques da Base

- > Período coberto: Setembro/2016 a Setembro/2018
- > Pedidos únicos: ~98 mil
- > Itens vendidos: ~112 mil
- > Categorias de produto: 70+ categorias (ex.: cama_mesa_banho, beleza_saude, esporte_lazer)
- > Clientes e Sellers: abrangem praticamente todos os estados do Brasil, com concentração em São Paulo (SP).
