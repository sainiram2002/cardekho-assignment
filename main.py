import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel
import ollama  # Use Ollama instead of OpenAI
import io
from PIL import Image

# Global Variable for DataFrame
df = None

# File Upload Function with Encoding Handling
def upload_csv(file):
    global df
    try:
        df = pd.read_csv(file.name, encoding="utf-8", encoding_errors="replace")  # Handles unknown characters
        return f"CSV uploaded successfully! Columns: {', '.join(df.columns)}"
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file.name, encoding="ISO-8859-1")  # Try an alternative encoding
            return f"CSV uploaded successfully with ISO-8859-1 encoding! Columns: {', '.join(df.columns)}"
        except Exception as e:
            return f"Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

# Query Processing Class
class QueryInput(BaseModel):
    query: str

# Question Answering Function
def answer_query(query: str):
    global df
    if df is None:
        return "Please upload a CSV file first."
    
    try:
        # Try handling structured queries using pandas
        if any(word in query.lower() for word in ["show", "find", "list", "get", "value", "count", "sum", "average", "mean", "max", "min"]):
            try:
                result = df.query(query)
                return result.to_string() if not result.empty else "No matching records found."
            except Exception:
                pass  # If pandas query fails, fall back to LLM
        
        # Process complex queries with LLM
        prompt = f"Analyze the following dataset and answer the query: {query}\nDataset:\n{df.head(10).to_string()}"
        response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
    except Exception as e:
        return f"Error processing query: {str(e)}"

# Graph Plotting Function
def plot_graph(x_column: str, y_column: str):
    global df
    if df is None:
        return "Please upload a CSV file first."
    
    try:
        if x_column not in df.columns or y_column not in df.columns:
            return f"Error: One or both columns '{x_column}' and '{y_column}' not found in dataset."

        # Drop NaN values to avoid plotting issues
        df_clean = df[[x_column, y_column]].dropna()

        if df_clean.empty:
            return "Error: No valid data available for plotting after removing NaN values."

        plt.figure(figsize=(8, 5))  # Set figure size

        # Convert non-numeric X-axis to categorical codes
        if not pd.api.types.is_numeric_dtype(df_clean[x_column]):
            df_clean[x_column] = df_clean[x_column].astype("category").cat.codes
        
        # Ensure Y-axis is numeric
        if not pd.api.types.is_numeric_dtype(df_clean[y_column]):
            return f"Error: Column '{y_column}' must be numeric for plotting."
        
        df_clean.plot(x=x_column, y=y_column, kind='line', marker='o', linestyle='-', color='b')

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight')  # Ensure the plot is properly saved
        plt.close()  # Close the figure to free memory
        img_buf.seek(0)

        # Convert BytesIO to PIL Image
        image = Image.open(img_buf)
        return image
    except Exception as e:
        return f"Error generating graph: {str(e)}"

# Gradio Interface
demo = gr.Interface(
    fn=upload_csv,
    inputs=gr.File(label="Upload CSV"),
    outputs="text",
    live=True
)

demo2 = gr.Interface(
    fn=answer_query,
    inputs=gr.Textbox(label="Enter Query"),
    outputs="text"
)

demo3 = gr.Interface(
    fn=plot_graph,
    inputs=[
        gr.Textbox(label="X-axis Column"),
        gr.Textbox(label="Y-axis Column")
    ],
    outputs="image"
)

app = gr.TabbedInterface([demo, demo2, demo3], ["Upload CSV", "Ask Questions", "Generate Graph"])

if __name__ == "__main__":
    app.launch()