# 🎯 Guide d'Intégration Zed - Reconnaissance Automatique du Workspace

Ce guide explique comment utiliser concrètement le proxy Ollama-Claude avec Zed pour que Claude Code travaille automatiquement dans le bon dossier de projet.

## 🚀 Solutions Pratiques

### **Option 1 : Helper Script (Solution Immédiate)**

Le script `zed-workspace-helper.py` automatise la détection et configuration du workspace.

#### Installation rapide :
```bash
# Dans le dossier du proxy
python3 zed-workspace-helper.py detect
```

#### Utilisation avant d'ouvrir Zed :
```bash
# 1. Démarrer le proxy
python3 main.py &

# 2. Naviguer vers votre projet
cd /path/to/your/project

# 3. Configurer le workspace automatiquement
python3 /path/to/proxy/zed-workspace-helper.py detect

# 4. Ouvrir Zed
zed .
```

#### Monitoring automatique (expérimental) :
```bash
# Surveille les changements de workspace
python3 zed-workspace-helper.py monitor
```

### **Option 2 : Configuration Manuelle par Projet**

Pour chaque projet Zed, vous pouvez configurer le workspace explicitement :

```bash
curl -X POST http://localhost:11435/api/set-working-directory \
  -H "Content-Type: application/json" \
  -d '{"working_dir": "/Users/arthur/Desktop/mon-projet"}'
```

### **Option 3 : Alias Shell (Solution Élégante)**

Créez un alias pour automatiser le processus :

```bash
# Ajoutez à votre ~/.zshrc ou ~/.bashrc
alias zed-claude='python3 /path/to/proxy/zed-workspace-helper.py detect && zed'

# Utilisation
cd /path/to/project
zed-claude .
```

### **Option 4 : Script de Lancement Personnalisé**

Créez un script qui combine tout :

```bash
#!/bin/bash
# ~/bin/zed-with-claude

PROXY_DIR="/Users/arthur/Desktop/Plugins/ollama-claude-code"
PROJECT_DIR="$1"

if [ -z "$PROJECT_DIR" ]; then
    PROJECT_DIR="$(pwd)"
fi

echo "🚀 Démarrage Zed avec Claude pour: $PROJECT_DIR"

# 1. S'assurer que le proxy tourne
if ! curl -s http://localhost:11435/health > /dev/null; then
    echo "📡 Démarrage du proxy..."
    cd "$PROXY_DIR" && python3 main.py &
    sleep 3
fi

# 2. Configurer le workspace
cd "$PROJECT_DIR"
python3 "$PROXY_DIR/zed-workspace-helper.py" detect

# 3. Lancer Zed
echo "✅ Ouverture de Zed..."
zed "$PROJECT_DIR"
```

## 🔧 Configuration Zed

Dans Zed, assurez-vous d'avoir la configuration suivante :

### `~/.config/zed/settings.json`
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

## 🎯 Workflow Recommandé

### Workflow Quotidien :
```bash
# 1. Démarrer le proxy (une fois par session)
cd ~/Desktop/Plugins/ollama-claude-code
python3 main.py &

# 2. Pour chaque projet
cd /path/to/your/project
python3 ~/Desktop/Plugins/ollama-claude-code/zed-workspace-helper.py detect
zed .

# 3. Claude Code travaille maintenant dans le bon dossier !
```

### Vérification :
```bash
# Vérifier que le workspace est bien configuré
python3 zed-workspace-helper.py status

# Tester dans Zed Agent Panel
"Quel est mon dossier de travail actuel ? Liste les fichiers."
```

## 🔍 Détection Automatique

Le proxy détecte automatiquement le workspace via :

1. **Headers HTTP** : `x-project-path`, `x-workspace`, etc.
2. **Patterns dans les prompts** : "working in", "project", "workspace"
3. **User-Agent** : Patterns comme `Zed/workspace=/path`
4. **Détection intelligente** : Recherche de `.git`, `package.json`, etc.

## 🐛 Troubleshooting

### Le workspace n'est pas détecté :
```bash
# Vérifier l'état du proxy
curl http://localhost:11435/api/sessions

# Forcer la configuration
python3 zed-workspace-helper.py set /path/to/correct/project

# Vérifier les logs du proxy
# (regarder les logs dans le terminal où tourne main.py)
```

### Claude Code ne voit pas les bons fichiers :
```bash
# Dans Zed Agent Panel, tester :
"Dans quel dossier travailles-tu ? Utilise l'outil LS pour lister les fichiers."

# Si incorrect, reconfigurer :
python3 zed-workspace-helper.py detect
```

## 🚀 Améliorations Futures

- **Extension Zed native** : Intégration directe dans Zed
- **Auto-détection temps réel** : Changement automatique lors du switch de projet
- **Configuration par projet** : Sauvegarde des préférences par workspace

## 📝 Commandes Utiles

```bash
# Helper script
python3 zed-workspace-helper.py detect    # Auto-détection
python3 zed-workspace-helper.py list      # Liste workspaces
python3 zed-workspace-helper.py status    # État du proxy
python3 zed-workspace-helper.py monitor   # Surveillance continue

# API directe
curl http://localhost:11435/                           # Info du proxy
curl http://localhost:11435/api/sessions               # Sessions actives
curl -X POST http://localhost:11435/api/set-working-directory \
  -d '{"working_dir": "/path"}'                         # Définir workspace
```

---

Avec ces solutions, Claude Code travaillera automatiquement dans le bon dossier de projet Zed ! 🎉