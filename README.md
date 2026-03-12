# RAG AGENT
A Retrieval-Augmented Generation (RAG) Agent that combines large language models like Ollama and Gemma models with external data source from Qdrant retrieval to provide accurate, context-aware responses.

## Steps to setup RAG Agent
### Install dependency
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu

### Run Ollama and Qdrant from docker 
sudo docker compose up -d --build
sudo docker compose up -d

Download gemma2 model from https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/blob/main/gemma-2-9b-it-Q8_0.gguf, \
wget --header="Authorization: Bearer (TOKEN)" \
  "https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q8_0.gguf" \
  -c -O gemma-2-9b-it-Q8_0.gguf

Create Modelfile.gemma2,

    cat > Modelfile.gemma2 << 'EOF'
    FROM ./gemma-2-9b-it-Q8_0.gguf

    TEMPLATE """<start_of_turn>user
    {{ if .System }}{{ .System }}<end_of_turn>
    <start_of_turn>model
    {{ end }}{{ .Prompt }}<end_of_turn>
    <start_of_turn>model
    {{ .Response }}<end_of_turn>"""

    PARAMETER num_ctx 8192
    PARAMETER temperature 0.3
    PARAMETER stop "<start_of_turn>"
    PARAMETER stop "<end_of_turn>"
    SYSTEM """You are a precise assistant that answers only using the provided JSON context."""
    EOF

sudo docker exec ragservice-ollama-1 mkdir -p /root/model-import

sudo docker cp Modelfile.gemma2 ragservice-ollama-1:/root/model-import/

sudo docker cp /root/ollama/gemma-2-9b-it-Q8_0.gguf ragservice-ollama-1:/root/model-import/gemma-2-9b-it-Q8_0.gguf

sudo docker exec -w /root/model-import ragservice-ollama-1 ollama create gemma2:9b -f Modelfile.gemma2

## Run application
uvicorn rag_service:app --host 0.0.0.0 --port 8001\
go run main.go