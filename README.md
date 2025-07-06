# 🚀 Ollama-to-Claude Proxy

Un serveur proxy FastAPI qui imite l'API Ollama mais route les requêtes vers Claude Code CLI, optimisé pour l'intégration avec Zed.

## 📋 Fonctionnalités

- **API compatible Ollama** : Endpoints `/api/chat`, `/api/generate`, `/api/tags`
- **Streaming en temps réel** : Réponses mot par mot avec Server-Sent Events
- **Gestion native des sessions** : Utilise les flags `-c` et `--resume` de Claude CLI
- **Détection automatique de workspace** : Via `cwd` et headers multiples
- **Optimisé pour les ressources** : Utilise le modèle par défaut (pas de thinking forcé)

## 🔧 Installation

```bash
git clone https://github.com/ArthurFranckPat/ollama-claude-code.git
cd ollama-claude-code
pip install fastapi uvicorn requests
```

## 🚀 Démarrage

```bash
# Démarrer le serveur proxy
python3 main.py

# Le serveur écoute sur http://localhost:11435
```

## 🎯 Intégration Zed

### Configuration Zed
Dans `~/.config/zed/settings.json` :
```json
{
  "language_models": {
    "ollama": {
      "api_url": "http://localhost:11435",
      "low_speed_timeout_in_seconds": 30
    }
  },
  "available_models": [
    {
      "provider": "ollama",
      "name": "claude-sonnet-4",
      "max_tokens": 8192
    }
  ]
}
```

### Workflow Simple

1. **Démarrer le proxy** (une fois par session)
   ```bash
   cd ollama-claude-code
   python3 main.py &
   ```

2. **Configurer le workspace** (par projet)
   ```bash
   cd /path/to/your/project
   python3 /path/to/proxy/zed-workspace-helper.py detect
   ```

3. **Ouvrir Zed**
   ```bash
   zed .
   ```

4. **Claude Code travaille maintenant dans le bon dossier !**

## 🎯 Avantages de cette Approche

### ✅ Efficacité des Ressources
- **Modèle par défaut** : Pas de forçage Sonnet 4 avec thinking
- **Simple `cwd`** : Pas besoin de `--add-dir` 
- **Moins de tokens** : Pas de surcharge thinking
- **Plus rapide** : Réponses plus directes

### ✅ Simplicité
- **Native Claude CLI** : Utilise les capacités natives du CLI
- **Pas de surcharge** : Juste `cd` vers le workspace
- **Configuration minimale** : Workspace détecté automatiquement

### ✅ Flexibilité
- **Auto-détection** : Via headers, prompts, et patterns
- **Fallback intelligent** : Toujours un workspace valide
- **Compatible Zed** : Intégration transparente

## 🔧 Scripts Utiles

### Helper de Workspace
```bash
# Auto-détection du workspace actuel
python3 zed-workspace-helper.py detect

# Lister les workspaces potentiels
python3 zed-workspace-helper.py list

# Définir un workspace spécifique
python3 zed-workspace-helper.py set /path/to/project

# Vérifier le statut du proxy
python3 zed-workspace-helper.py status
```

### Tests
```bash
# Tester les fonctionnalités
python3 test_workspace.py
```

## 📊 API Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/api/chat` | POST | Chat avec streaming |
| `/api/generate` | POST | Génération simple |
| `/api/tags` | GET | Liste des modèles |
| `/api/set-working-directory` | POST | Définir workspace |
| `/api/get-working-directory` | GET | Obtenir workspace |
| `/api/sessions` | GET | Sessions actives |
| `/health` | GET | Santé du serveur |

## 🐛 Troubleshooting

### Claude ne voit pas les bons fichiers
```bash
# Vérifier le workspace actuel
python3 zed-workspace-helper.py status

# Reconfigurer si nécessaire
python3 zed-workspace-helper.py detect
```

### Le proxy ne répond pas
```bash
# Vérifier que le proxy tourne
curl http://localhost:11435/health

# Redémarrer si nécessaire
pkill -f "python3 main.py"
python3 main.py &
```

## 📈 Optimisations Futures

- **Extension Zed native** : Intégration directe sans proxy
- **Cache intelligent** : Réutilisation des sessions
- **Détection temps réel** : Changement automatique de workspace
- **Métriques** : Monitoring des performances

## 🤝 Contribution

Les contributions sont les bienvenues ! Ouvrez une issue ou une pull request.

## 📄 License

MIT License - voir le fichier LICENSE pour plus de détails.