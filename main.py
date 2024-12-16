import os
import re
import json
import requests
import nbformat as nbf
import matplotlib.pyplot as plt
import seaborn as sns
from github import Github
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Any, Optional


class AIModelType(Enum):
    GEMINI = auto()
    META_AI = auto()
    CLAUDE = auto()


@dataclass
class AIModelConfig:
    """Configuration for different AI models"""
    model_type: AIModelType
    api_key: str
    base_url: str
    headers: Dict[str, str]


class GitHubDocumentationGenerator:
    def __init__(self, repo_url, github_token):
        """
        Initialize the documentation generator with GitHub repository details
        """
        self.repo_url = repo_url
        self.github_token = github_token

        # Extract owner and repo name
        match = re.search(r'github\.com/([^/]+)/([^/]+)', repo_url)
        if not match:
            raise ValueError("Invalid GitHub repository URL")

        self.owner = match.group(1)
        self.repo_name = match.group(2)

        # Initialize GitHub connection
        self.g = Github(self.github_token)
        self.repo = self.g.get_repo(f"{self.owner}/{self.repo_name}")

    def analyze_repo_structure(self):
        """Analyze the repository structure"""
        contents = self.repo.get_contents("")
        repo_structure = {
            'total_files': 0,
            'file_types': {},
            'directories': []
        }

        def traverse_contents(contents_list, current_path=''):
            for content in contents_list:
                if content.type == 'dir':
                    repo_structure['directories'].append(current_path + content.name)
                    sub_contents = self.repo.get_contents(content.path)
                    traverse_contents(sub_contents, current_path + content.name + '/')
                else:
                    repo_structure['total_files'] += 1
                    file_ext = os.path.splitext(content.name)[1]
                    repo_structure['file_types'][file_ext] = repo_structure['file_types'].get(file_ext, 0) + 1

        traverse_contents(contents)
        return repo_structure

    def generate_visualization(self, repo_structure):
        """Create visualizations of repository structure"""
        plt.figure(figsize=(10, 6))
        file_types = repo_structure['file_types']

        sns.barplot(x=list(file_types.keys()), y=list(file_types.values()))
        plt.title(f"File Types in {self.repo_name}")
        plt.xlabel("File Extensions")
        plt.ylabel("Count")
        plt.xticks(rotation=45)

        visualization_path = f"{self.repo_name}_file_types.png"
        plt.tight_layout()
        plt.savefig(visualization_path)
        plt.close()

        return visualization_path

    def generate_jupyter_documentation(self, repo_structure):
        """Generate a comprehensive Jupyter notebook documentation"""
        nb = nbf.v4.new_notebook()

        # Title and Overview
        nb['cells'] = [
            nbf.v4.new_markdown_cell(f"# {self.repo_name} Documentation\n\n"
                                     f"## Repository Overview\n\n"
                                     f"**Total Files**: {repo_structure['total_files']}\n\n"
                                     f"**Directories**: {', '.join(repo_structure['directories'])}\n\n"
                                     "## File Type Distribution"),

            nbf.v4.new_code_cell([
                "import matplotlib.pyplot as plt\n"
                "import seaborn as sns\n"
                f"file_types = {repr(repo_structure['file_types'])}\n"
                "plt.figure(figsize=(10, 6))\n"
                "sns.barplot(x=list(file_types.keys()), y=list(file_types.values()))\n"
                "plt.title('File Types in Repository')\n"
                "plt.xlabel('File Extensions')\n"
                "plt.ylabel('Count')\n"
                "plt.xticks(rotation=45)\n"
                "plt.tight_layout()\n"
                "plt.show()"
            ]),

            nbf.v4.new_markdown_cell("## Installation Instructions\n\n"
                                     f"To install and set up the {self.repo_name} project:\n\n"
                                     "```bash\n"
                                     f"git clone {self.repo_url}\n"
                                     f"cd {self.repo_name}\n"
                                     "pip install -r requirements.txt\n"
                                     "```")
        ]

        return nb

    def create_documentation(self):
        """Main method to generate comprehensive documentation"""
        # Analyze repository
        repo_structure = self.analyze_repo_structure()

        # Generate visualization
        visualization_path = self.generate_visualization(repo_structure)

        # Generate Jupyter notebook documentation
        jupyter_doc = self.generate_jupyter_documentation(repo_structure)

        # Save Jupyter notebook
        with open(f"{self.repo_name}_documentation.ipynb", 'w') as f:
            nbf.write(jupyter_doc, f)

        return {
            'repo_structure': repo_structure,
            'visualization': visualization_path,
            'jupyter_notebook': f"{self.repo_name}_documentation.ipynb"
        }


class MultiAITaskResolver:
    def __init__(self, configs: Dict[AIModelType, AIModelConfig]):
        """Initialize MultiAI Task Resolver with model configurations"""
        self.configs = configs

    def resolve_task(self, task_description: str, preferred_model: AIModelType, fallback_model: Optional[AIModelType] = None):
        """Resolve tasks using preferred and fallback AI models"""
        try:
            result = self._call_ai_model(preferred_model, task_description)
            return {"model": preferred_model.name, "result": result, "status": "primary_model_success"}
        except Exception as primary_error:
            if fallback_model:
                try:
                    fallback_result = self._call_ai_model(fallback_model, task_description)
                    return {"model": fallback_model.name, "result": fallback_result, "status": "fallback_model_success"}
                except Exception as fallback_error:
                    raise RuntimeError(f"Fallback failed: {fallback_error}")
            raise RuntimeError(f"Primary failed: {primary_error}")

    def _call_ai_model(self, model_type: AIModelType, task: str) -> Dict[str, Any]:
        """Call specific AI model's API"""
        config = self.configs.get(model_type)
        if not config:
            raise ValueError(f"No configuration for {model_type}")

        payload = {"model": model_type.name, "task": task}
        response = requests.post(config.base_url, headers=config.headers, json=payload)
        response.raise_for_status()
        return response.json()

    def categorize_task(self, task_description: str) -> AIModelType:
        """Categorize task for optimal model selection"""
        logic_keywords = ["algorithm", "optimization", "logic"]
        creative_keywords = ["story", "design", "creative"]

        if any(keyword in task_description.lower() for keyword in logic_keywords):
            return AIModelType.GEMINI
        if any(keyword in task_description.lower() for keyword in creative_keywords):
            return AIModelType.META_AI
        return AIModelType.GEMINI


def main():
    GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
    REPO_URL = "https://github.com/username/repo"

    # GitHub Documentation
    doc_generator = GitHubDocumentationGenerator(REPO_URL, GITHUB_TOKEN)
    documentation = doc_generator.create_documentation()

    # AI Task Resolver
    configs = {
        AIModelType.GEMINI: AIModelConfig(
            model_type=AIModelType.GEMINI,
            api_key=os.getenv('GEMINI_API_KEY', ''),
            base_url="https://api.gemini.ai/v1/task",
            headers={"Authorization": f"Bearer {os.getenv('GEMINI_API_KEY', '')}"}
        )
    }
    resolver = MultiAITaskResolver(configs)

    # Example tasks
    logic_task = "Optimize an algorithm for sorting integers"
    creative_task = "Write a short story about AI emotions"

    logic_result = resolver.resolve_task(logic_task, AIModelType.GEMINI)
    creative_result = resolver.resolve_task(creative_task, AIModelType.META_AI)

    print("Documentation generated:", documentation)
    print("Logic task result:", logic_result)
    print("Creative task result:", creative_result)


if __name__ == "__main__":
    main()
