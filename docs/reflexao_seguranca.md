# Reflexão sobre boas práticas de software seguro

## Aplicações ao problema

Mesmo utilizando um dataset público, algumas práticas de segurança e privacidade continuam relevantes para a solução:

### 1. Minimização de dados
A aplicação deve solicitar somente os atributos necessários para a predição. Isso reduz superfície de exposição e simplifica o tratamento dos dados.

### 2. Validação de entrada
Os campos numéricos enviados pelo front-end são validados no back-end por meio do schema da requisição. Essa validação reduz risco de dados inválidos, falhas de processamento e abuso da API.

### 3. Evitar exposição desnecessária do modelo
O modelo é carregado no back-end e não fica exposto diretamente ao navegador. O front-end envia apenas os atributos necessários para a predição.

### 4. Anonimização quando houver dados reais
Se o problema envolvesse dados de clientes ou pacientes, seria importante remover identificadores diretos e indiretos, além de aplicar técnicas de anonimização ou pseudonimização antes do treinamento.

### 5. Controle de acesso
Em um cenário real, endpoints de predição e consulta de métricas devem exigir autenticação, autorização e trilha de auditoria.

### 6. Logs e monitoramento
Logs devem registrar erros técnicos, mas sem persistir dados sensíveis em claro. Também é recomendável monitorar volume, origem e padrão de uso da API.

### 7. Integridade do modelo
O arquivo do modelo deve ser versionado, validado antes da implantação e substituído apenas após passar pelos testes automatizados de desempenho definidos no projeto.

## Síntese

As práticas de software seguro ajudam a evitar vazamento de dados, uso indevido da API, implantação de modelos com qualidade insuficiente e falhas de operação. Mesmo em um MVP simples, validação de entrada, minimização de dados, controle de acesso e governança de artefatos já agregam valor real.
