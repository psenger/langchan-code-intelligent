from dataclasses import dataclass
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
import os
import glob
import hashlib
import pandas as pd
import lancedb
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough


@dataclass
class CodeDocument:
    """Data class for representing a code document"""
    id: str
    content: str
    file_path: str
    file_type: str


class DocumentReader:
    """Responsible for reading and processing code files"""

    def __init__(self, file_extensions: List[str]):
        self.file_extensions = file_extensions

    def read_directory(self, directory: str) -> List[CodeDocument]:
        """Recursively reads all code files from the given directory"""
        documents = []
        stats = {"total": 0, "processed": 0, "skipped": 0}

        print("\nScanning for files...")

        for ext in self.file_extensions:
            pattern = os.path.join(directory, f"**/*{ext}")
            files = glob.glob(pattern, recursive=True)
            stats["total"] += len(files)

            for file_path in files:
                if '__pycache__' in file_path or '.venv' in file_path or '.git' in file_path:
                    stats["skipped"] += 1
                    continue

                try:
                    print(f"Reading: {file_path}")
                    documents.append(self._process_file(file_path, directory, ext))
                    stats["processed"] += 1
                except Exception as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    stats["skipped"] += 1

        self._print_stats(stats)
        return documents

    def _process_file(self, file_path: str, directory: str, ext: str) -> CodeDocument:
        """Process a single file and return a CodeDocument"""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            relative_path = os.path.relpath(file_path, directory)
            return CodeDocument(
                id=self._generate_file_id(relative_path, content),
                content=content,
                file_path=relative_path,
                file_type=ext[1:]  # Remove the dot from extension
            )

    def _generate_file_id(self, file_path: str, content: str) -> str:
        """Generate a unique ID for a file"""
        return hashlib.sha256(f"{file_path}{content}".encode()).hexdigest()

    def _print_stats(self, stats: Dict[str, int]) -> None:
        """Print file processing statistics"""
        print(f"\nScan complete:")
        print(f"Total files found: {stats['total']}")
        print(f"Successfully processed: {stats['processed']}")
        print(f"Skipped/Failed: {stats['skipped']}")
        print("-" * 50)


class VectorDatabase:
    """Handles vector database operations"""

    def __init__(self, db_path: str, table_name: str, embeddings_model):
        self.db = lancedb.connect(db_path)
        self.table_name = table_name
        self.embeddings_model = embeddings_model

    def get_or_create_table(self):
        """Get existing table or create a new one"""
        import pyarrow as pa

        if self.table_name not in self.db.table_names():
            schema = pa.schema([
                ('id', pa.string()),
                ('content', pa.string()),
                ('file_path', pa.string()),
                ('file_type', pa.string()),
                ('vector', pa.list_(pa.float32(), 768))
            ])
            return self.db.create_table(self.table_name, schema=schema)
        return self.db.open_table(self.table_name)

    def store_documents(self, documents: List[CodeDocument], batch_size: int = 10):
        """Store documents with their embeddings"""
        if not documents:
            return

        total_documents = len(documents)
        total_batches = (total_documents + batch_size - 1) // batch_size  # Round up division

        print(f"\nProcessing {total_documents} documents in {total_batches} batches")
        print(f"Batch size: {batch_size}")
        print("-" * 50)

        table = self.get_or_create_table()

        for i in range(0, total_documents, batch_size):
            batch = documents[i:i + batch_size]
            current_batch = (i // batch_size) + 1

            print(f"\nBatch {current_batch}/{total_batches}")
            print(f"Processing documents {i + 1}-{min(i + batch_size, total_documents)} of {total_documents}")

            self._process_batch(batch, table)

            # Show progress percentage
            progress = (current_batch / total_batches) * 100
            print(f"Overall progress: {progress:.1f}%")

    def _process_batch(self, documents: List[CodeDocument], table):
        """Process a batch of documents"""
        print("  - Generating embeddings...")
        texts = [doc.content for doc in documents]
        embeddings = self.embeddings_model.embed_documents(texts)

        print("  - Creating data records...")
        data = [
            {
                "id": doc.id,
                "content": doc.content,
                "file_path": doc.file_path,
                "file_type": doc.file_type,
                "vector": vector
            }
            for doc, vector in zip(documents, embeddings)
        ]

        print("  - Storing in database...")
        table.add(pd.DataFrame(data))
        print("  - Batch complete")

    def search_similar(self, query_embedding: List[float], k: int = 5) -> pd.DataFrame:
        """Search for similar documents"""
        table = self.get_or_create_table()
        return table.search(query_embedding).limit(k).to_pandas()


class ChainFactory:
    """Factory for creating different types of LLM chains"""

    def __init__(self, llm):
        self.llm = llm
        self._chains = {
            "refactor": self._create_refactor_chain,
            "feature": self._create_feature_chain,
            "bug": self._create_bug_chain,
            "custom": self._create_custom_chain
        }

    def get_chain(self, action: str):
        """Get appropriate chain based on action"""
        chain_type = next(
            (key for key in self._chains.keys() if key in action.lower()),
            "custom"
        )
        return self._chains[chain_type]()

    def _create_refactor_chain(self):
        template = """You are an expert Python code refactoring assistant. Analyze the following code and suggest improvements.

        Code Context:
        {context}

        Specific Code to Refactor:
        {user_input}

        Consider and address the following aspects:
        1. Design Patterns - Identify applicable Python patterns that could improve the code structure
        2. Performance Implications - Note any performance concerns and optimization opportunities
        3. Maintenance Concerns - Highlight areas that might be difficult to maintain
        4. Code Duplication - Identify repeated patterns that could be abstracted
        5. Python Best Practices - Suggest improvements based on PEP 8 and Python idioms
        6. Type Hints - Opportunities to improve type annotations
        7. Error Handling - Proper exception handling patterns

        Provide your response in this format:
        1. Current Pattern Analysis
        2. Suggested Improvements
        3. Refactored Code
        4. Implementation Steps
        """
        return self._create_chain(template)

    def _create_feature_chain(self):
        template = """You are an expert Python developer. Suggest implementation for a new feature.

        Existing Code Context:
        {context}

        Feature Requirements:
        {user_input}

        Consider and address:
        1. Integration Points - How the feature fits with existing code
        2. State Management - Required state changes and management approach
        3. Error Handling - Comprehensive exception handling strategy
        4. Testing Considerations - Key test cases and testing approach using pytest
        5. Performance Implications - Any performance considerations
        6. Type Safety - Type hints and runtime type checking
        7. Documentation - Docstring requirements and examples

        Provide your response in this format:
        1. Implementation Strategy
        2. Required Changes
        3. New Code
        4. Integration Steps
        5. Testing Plan
        """
        return self._create_chain(template)

    def _create_bug_chain(self):
        template = """You are an expert Python debugger. Analyze the following code for potential bugs and issues.

        Code Context:
        {context}

        Specific Concern/Error:
        {user_input}

        Perform a thorough analysis considering:
        1. Error Patterns - Common Python error patterns
        2. Edge Cases - Potential edge cases that might cause issues
        3. Async Operations - Proper handling of async/await
        4. Resource Management - Context managers and cleanup
        5. Memory Usage - Memory leaks and garbage collection
        6. Type Safety - Type-related issues and runtime type checking
        7. Exception Handling - Proper exception handling patterns

        Provide your response in this format:
        1. Issue Analysis
        2. Root Cause
        3. Fix Recommendation
        4. Prevention Strategy
        """
        return self._create_chain(template)

    def _create_custom_chain(self):
        template = """You are an expert Python code assistant. Use the following code context to answer the question.
        If you cannot answer based on the context, say so.

        Code Context:
        {context}

        Question:
        {user_input}

        Provide a detailed, well-structured response that directly addresses the question.
        If code changes are needed, provide complete, working Python code snippets.
        Include type hints and docstrings where appropriate.
        """
        return self._create_chain(template)

    def _create_chain(self, template: str):
        """Create a chain with the given template"""
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "user_input"]
        )

        # Create a RunnableSequence instead of LLMChain
        chain = (
                {"context": RunnablePassthrough(), "user_input": RunnablePassthrough()}
                | prompt
                | self.llm
        )
        return chain


class CodeAssistant:
    """Main class for code assistance functionality"""

    def __init__(self, db_path: str = "code_embeddings.db"):
        # Initialize models
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text:latest",
            base_url="http://localhost:11434"
        )
        self.llm = OllamaLLM(
            model="codellama:13b",
            base_url="http://localhost:11434",
            temperature=0.2
        )

        # Initialize components
        self.file_extensions = ['.js', '.ts', '.jsx', '.tsx', '.test.js', '.test.ts', '.spec.js', '.spec.ts']
        self.document_reader = DocumentReader(self.file_extensions)
        self.vector_db = VectorDatabase(db_path, "code_embeddings", self.embeddings)
        self.chain_factory = ChainFactory(self.llm)

    def get_directory(self) -> str:
        """Get and validate directory from user input"""
        while True:
            directory = input("\nEnter the path to your code directory (or '.' for current directory): ").strip()

            # Convert relative path to absolute path
            directory = os.path.abspath(directory)

            if os.path.isdir(directory):
                return directory
            else:
                print(f"Invalid directory path: {directory}")
                retry = input("Would you like to try again? (y/n): ").lower()
                if retry != 'y':
                    return ""

    def process_directory(self, directory: str) -> None:
        """Process all files in a directory"""
        documents = self.document_reader.read_directory(directory)
        if documents:
            print(f"Found {len(documents)} code files. Creating embeddings...")
            self.vector_db.store_documents(documents)
            print("Embeddings created successfully!")
        else:
            print("No code files found in the selected directory!")

    def process_request(self, action: str, user_input: str = "") -> str:
        """Process a user request"""
        try:
            context_docs = self._get_relevant_context(action, user_input)
            context = self._format_context(context_docs)
            chain = self.chain_factory.get_chain(action)

            # Use invoke instead of run
            return chain.invoke({
                "context": self._create_context_summary(context_docs) + "\n\n" + context,
                "user_input": user_input
            })
        except Exception as e:
            return self._format_error_response(str(e))

    def _get_relevant_context(self, query: str, user_input: str = "", k: int = 5) -> List[Dict]:
        """Get relevant context for a query"""
        search_terms = []
        if user_input:
            search_terms.extend(self._extract_code_terms(user_input))
        if query:
            search_terms.extend(self._extract_code_terms(query))

        search_query = " ".join(search_terms) if search_terms else query
        query_embedding = self.embeddings.embed_query(search_query)

        results = self.vector_db.search_similar(query_embedding, k)
        return self._format_search_results(results)

    def _format_context(self, context_docs: List[Dict]) -> str:
        """Format context documents for output"""
        grouped_docs = {}
        for doc in context_docs:
            doc_type = doc['metadata']['type']
            if doc_type not in grouped_docs:
                grouped_docs[doc_type] = []
            grouped_docs[doc_type].append(doc)

        formatted_sections = []

        # Format implementation files
        for ext in ['ts', 'js', 'tsx', 'jsx']:
            if ext in grouped_docs:
                formatted_sections.extend(self._format_section(ext, grouped_docs[ext]))

        # Format test files
        for ext in ['test.ts', 'test.js', 'spec.ts', 'spec.js']:
            if ext in grouped_docs:
                formatted_sections.extend(self._format_section(ext, grouped_docs[ext]))

        return "\n\n".join(formatted_sections)

    def _format_section(self, ext: str, docs: List[Dict]) -> List[str]:
        """Format a section of documents"""
        section = [f"\n--- {ext.upper()} Implementation Files ---"]
        for doc in docs:
            section.append(
                f"File: {doc['metadata']['path']}\n"
                f"```{doc['metadata']['type']}\n{doc['page_content']}```"
            )
        return section

    def _create_context_summary(self, context_docs: List[Dict]) -> str:
        """Create a summary of the context"""
        return "\n".join([
            "Context Summary:",
            f"- Number of files: {len(context_docs)}",
            "- Files included:",
            *[f"  * {doc['metadata']['path']} ({doc['metadata']['type']})"
              for doc in context_docs]
        ])

    def _format_search_results(self, results: pd.DataFrame) -> List[Dict]:
        """Format search results"""
        return [{
            "page_content": row["content"],
            "metadata": {
                "path": row["file_path"],
                "type": row["file_type"]
            }
        } for _, row in results.iterrows()]

    def _extract_code_terms(self, text: str) -> List[str]:
        """Extract code-related terms from text"""
        common_words = {'the', 'and', 'or', 'analyze', 'suggest', 'code', 'improve'}
        return [
            word for word in text.replace('.', ' ').replace('_', ' ').split()
            if len(cleaned_word := word.lower().strip()) > 2
               and cleaned_word not in common_words
               and not cleaned_word.startswith(('the', 'and', 'for'))
        ]

    def _format_error_response(self, error: str) -> str:
        """Format error response"""
        return f"""I apologize, but I encountered an issue while processing your request: {error}

Please try to:
1. Provide more specific details about what you want to analyze
2. Specify the exact file or component you're interested in
3. Break down complex requests into smaller parts"""


def main():
    """Main entry point"""
    assistant = CodeAssistant()

    # Get directory from user
    print("Please select the directory containing your code...")
    directory = assistant.get_directory()

    if not directory:
        print("No directory selected. Exiting...")
        return

    print(f"\nScanning directory: {directory}")
    assistant.process_directory(directory)

    while True:
        print("\nWhat would you like to do?")
        print("1. Code Refactoring")
        print("2. Feature Addition")
        print("3. Bug Analysis")
        print("4. Custom Query")
        print("5. Exit")

        choice = input("\nEnter your choice (1-5): ")

        if choice == "5":
            break

        action_map = {
            "1": "Analyze the code and suggest refactoring improvements",
            "2": "Suggest implementation for a new feature",
            "3": "Analyze the code for potential bugs and suggest fixes",
            "4": "Custom query"
        }

        if choice not in action_map:
            print("Invalid choice!")
            continue

        action = action_map[choice]
        if choice == "4":
            action = input("\nEnter your custom query: ")

        user_input = ""
        if choice in ["1", "2"]:
            user_input = input("\nPlease provide any additional code or requirements: ")


        print("\nProcessing your request...")
        response = assistant.process_request(action, user_input)

        print("\nResponse:")
        print(response)

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()