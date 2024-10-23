# Gen-AI-ToolKit

To run this script, you need to have Python installed on your system. You can download Python from [python.org](https://www.python.org/downloads/).

### Steps

1. Clone this repository:

   ```bash
   git clone https://github.com/puja-chaudhury/Gen-AI-ToolKit.git
   cd Gen-AI-ToolKit
   
2.Download Ollama from https://github.com/ollama/ollama?tab=readme-ov-file

  ```
  ollama run llama3.2:3b


3. Create python environment:
   
     ```
   mkvirtualenv env_name
   workon env_name

4. Install the required dependencies:

   ```
   pip install -r requirements.txt
   
5. Create your own TAVILY_API_KEY by visiting : https://app.tavily.com/home and replace it in .env file.

6. Run each AI program:

   ```
    streamlit run simple_text_generation.py
    streamlit run rag_translation.py
    streamlit run corrective_rag.py
