#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Question Answering Prompt Definition
Used to answer user questions based on memory records

Optimization Notes:
- Optimized for LoCoMo dataset evaluation criteria
- Supports multiple question types: temporal, factual, multi-hop reasoning, open-domain reasoning
- Ensures concise, direct output format for easier F1 score evaluation
"""

# QA系统的System Prompt
QA_SYSTEM_PROMPT = """You are a precise question-answering assistant. Your task is to answer questions based solely on the provided memory records.

CRITICAL OUTPUT RULES:
1. Provide DIRECT, CONCISE answers - use minimal words
2. For factual questions: give exact facts (names, dates, places, events)
3. For list questions: use comma-separated format (e.g., "item1, item2, item3")
4. For time/date questions: provide specific dates or time periods mentioned
5. For reasoning questions: give brief explanations only when necessary
6. If information is NOT in the memories: respond EXACTLY with "No information available"
7. DO NOT add extra explanations, greetings, or conversational filler
8. DO NOT make assumptions beyond what's explicitly stated in the memories"""

def build_qa_prompt(memory_context: str, question: str) -> str:
    """
    Build the user prompt for question answering
    
    Args:
        memory_context: Memory records context
        question: User question
        
    Returns:
        Complete QA Prompt
        
    Examples:
        Q: "What books has she read?" 
        A: "Book1, Book2, Book3"
        
        Q: "When did she go to the museum?"
        A: "5 July 2023"
        
        Q: "What is her profession?"
        A: "Software engineer"
    """
    return f"""Memory Records:
{memory_context}

Question: {question}

Instructions:
- Extract the answer DIRECTLY from the memory records above
- Answer using the SHORTEST possible form
- For multiple items, use comma-separated format
- For dates/times, provide exact information from memories
- If the memories do not contain the answer, respond with: "No information available"
- DO NOT add explanations unless the question explicitly asks for reasoning

Answer:"""
