# MVP de Classificação de Vinhos

Projeto completo do MVP com:
- notebook em formato Colab;
- treinamento e exportação do modelo;
- aplicação full stack simples com FastAPI + interface web;
- teste automatizado com PyTest baseado em métricas e limiares;
- reflexão de segurança aplicada ao problema.

## Estrutura

- `notebooks/mvp_classificacao_vinhos.ipynb`: notebook principal.
- `backend/`: API, carregamento do modelo, testes e ativos web.
- `artifacts/`: modelo treinado e metadados.
- `requirements.txt`: dependências do projeto.

## Dataset

O notebook carrega o dataset por URL pública:
- UCI Wine Data: `https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data`

No notebook, os nomes das colunas são definidos explicitamente para permitir execução direta no Google Colab.

## Como executar localmente

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate # Windows

pip install -r requirements.txt
python train_model.py
uvicorn backend.main:app --reload
```

A aplicação ficará disponível em:
- `http://127.0.0.1:8000`

## Como rodar os testes

```bash
pytest -q
```

## Observação sobre o notebook

O notebook foi preparado para execução do início ao fim no Google Colab. Ao final da execução, ele exporta:
- `wine_classifier.joblib`
- `model_metadata.json`

Esses arquivos também já acompanham o projeto na pasta `artifacts/`.
