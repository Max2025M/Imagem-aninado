#!/bin/bash
# Script de deploy rápido para Render Free
# Cria pastas, instala dependências e roda Docker

echo "Preparando pastas temporárias..."
mkdir -p uploads outputs

echo "Construindo imagem Docker..."
docker build -t avatar_audio_app .

echo "Rodando container Docker na porta 5000..."
docker run -d -p 5000:5000 --name avatar_audio_container avatar_audio_app

echo "✅ Projeto pronto! Acesse http://localhost:5000"
