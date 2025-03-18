"""
Agent Evaluator

This module provides tools for evaluating agent performance based on rubrics
and interaction data, supporting agent evolution and team composition.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Ensure we can import the repository
try:
    from clubhouse.agents.repositories.agent_repository import AgentRepository, AgentPerformanceMetrics
except ImportError:
    # Handle the import error gracefully for testing
    AgentRepository = object
    AgentPerformanceMetrics = object


class EvaluationRubric:
    """Evaluation rubric for assessing agent performance."""
    
    def __init__(self):
        # Context preservation evaluation
        self.context_rules = [
            (r'\bson\b.*\bdaughter\b|\bdaughter\b.*\bson\b', -0.5, "Gender confusion"),
            (r'\b(\d+)[^\d\n]{1,30}\1\b', -0.3, "Numeric detail inconsistency"),
            (r'\b([a-zA-Z]+)[^a-zA-Z\n]{1,30}\1s?\b|\bs?([a-zA-Z]+)[^a-zA-Z\n]{1,30}\2\b', 0.2, "Consistent detail reference")
        ]
        
        # Reference quality evaluation
        self.reference_rules = [
            (r'(?:according to|as ([A-Z][a-z]+ )?mentioned)', 0.2, "Attribution to source"),
            (r'([A-Z][a-z]+)(?:\'s suggestion|\'s point|\'s idea)', 0.3, "Named reference to collaborator"),
            (r'(building on|extending) ([A-Z][a-z]+)\'s', 0.3, "Building on collaborator's ideas")
        ]
        
        # Pragmatism evaluation
        self.pragmatism_rules = [
            (r'(?:first|step 1|begin by)[^.!?]*', 0.2, "Clear first steps"),
            (r'(?:specific|exact|precise)[^.!?]*(?:brand|model|version|type)', 0.3, "Specific recommendations"),
            (r'(?:approximate|around|\$\d+|\d+ dollars)', 0.2, "Price information")
        ]
    
    def evaluate_context_preservation(self, problem: str, solution: str) -> Tuple[float, List[str]]:
        """
        Evaluate how well the solution preserves context from the problem.
        
        Args:
            problem: Original problem statement
            solution: Proposed solution
            
        Returns:
            Score (0.0-1.0) and explanation list
        """
        score = 0.7  # Start with a default score
        explanations = []
        
        # Check for key context elements
        key_elements = self._extract_key_elements(problem)
        
        # Check each key element exists in the solution
        for element, element_type in key_elements.items():
            if element.lower() in solution.lower():
                score += 0.05
                explanations.append(f"Preserved {element_type} '{element}'")
            else:
                score -= 0.1
                explanations.append(f"Failed to preserve {element_type} '{element}'")
        
        # Apply context rules
        for pattern, score_mod, explanation in self.context_rules:
            if re.search(pattern, solution, re.IGNORECASE):
                if score_mod < 0:
                    explanations.append(f"Issue: {explanation}")
                else:
                    explanations.append(f"Strength: {explanation}")
                score += score_mod
        
        # Ensure score is within bounds
        score = max(0.0, min(1.0, score))
        
        return score, explanations
    
    def evaluate_reference_quality(self, discussion: List[Dict], agent_name: str, agent_response: str) -> Tuple[float, List[str]]:
        """
        Evaluate how well an agent references and builds on others' contributions.
        
        Args:
            discussion: Full discussion history
            agent_name: Name of the agent being evaluated
            agent_response: The agent's response
            
        Returns:
            Score (0.0-1.0) and explanation list
        """
        score = 0.5  # Start with a neutral score
        explanations = []
        
        # Get names of other agents in the discussion
        other_agents = set()
        for entry in discussion:
            if entry.get("sender") and entry["sender"] != agent_name:
                other_agents.add(entry["sender"])
        
        # Check for references to other agents
        referenced_agents = set()
        for agent in other_agents:
            if agent in agent_response:
                referenced_agents.add(agent)
                score += 0.1
                explanations.append(f"Referenced collaborator '{agent}'")
        
        # Apply reference rules
        for pattern, score_mod, explanation in self.reference_rules:
            if re.search(pattern, agent_response, re.IGNORECASE):
                score += score_mod
                explanations.append(f"Strength: {explanation}")
        
        # Penalize lack of references if there were other agents to reference
        if not referenced_agents and other_agents:
            score -= 0.2
            explanations.append("Did not reference any collaborators")
        
        # Ensure score is within bounds
        score = max(0.0, min(1.0, score))
        
        return score, explanations
    
    def evaluate_pragmatism(self, solution: str) -> Tuple[float, List[str]]:
        """
        Evaluate the pragmatism and actionability of a solution.
        
        Args:
            solution: Proposed solution
            
        Returns:
            Score (0.0-1.0) and explanation list
        """
        score = 0.4  # Start with a below-average score
        explanations = []
        
        # Apply pragmatism rules
        for pattern, score_mod, explanation in self.pragmatism_rules:
            if re.search(pattern, solution, re.IGNORECASE):
                score += score_mod
                explanations.append(f"Strength: {explanation}")
        
        # Check for actionable steps
        steps_count = len(re.findall(r'(?:first|second|third|next|finally|lastly)[^.!?]*[.!?]', solution, re.IGNORECASE))
        if steps_count >= 3:
            score += 0.2
            explanations.append(f"Provides {steps_count} clear action steps")
        elif steps_count > 0:
            score += 0.1
            explanations.append(f"Provides {steps_count} action steps")
        else:
            score -= 0.1
            explanations.append("Lacks clear action steps")
        
        # Check for concrete suggestions
        concrete_count = len(re.findall(r'(?:recommend|suggest)[^.!?]*(?:specific|particular)[^.!?]*[.!?]', solution, re.IGNORECASE))
        if concrete_count > 0:
            score += 0.1 * concrete_count
            explanations.append(f"Provides {concrete_count} concrete suggestions")
        
        # Ensure score is within bounds
        score = max(0.0, min(1.0, score))
        
        return score, explanations
    
    def _extract_key_elements(self, text: str) -> Dict[str, str]:
        """Extract key context elements from a text."""
        elements = {}
        
        # Extract named entities (simple approach)
        name_matches = re.findall(r'\b([A-Z][a-z]+)\b', text)
        for name in name_matches:
            elements[name] = "name"
        
        # Extract gender markers
        if re.search(r'\bson\b', text, re.IGNORECASE):
            elements["son"] = "gender reference"
        if re.search(r'\bdaughter\b', text, re.IGNORECASE):
            elements["daughter"] = "gender reference" 
        if re.search(r'\bboy\b', text, re.IGNORECASE):
            elements["boy"] = "gender reference"
        if re.search(r'\bgirl\b', text, re.IGNORECASE):
            elements["girl"] = "gender reference"
        
        # Extract age information
        age_matches = re.findall(r'\b(\d+)[- ](?:year|yr)(?:s)?[- ]old\b', text, re.IGNORECASE)
        age_matches.extend(re.findall(r'\bage (\d+)\b', text, re.IGNORECASE))
        for age in age_matches:
            elements[age] = "age"
        
        # Extract interests
        interest_matches = re.findall(r'(?:interested in|likes|loves|enjoys) ([^,.!?]+)', text, re.IGNORECASE)
        for i, interest in enumerate(interest_matches):
            elements[interest.strip()] = f"interest_{i+1}"
        
        return elements


class AgentEvaluator:
    """Evaluator for agent performance in discussions and problem-solving."""
    
    def __init__(self, repository: Optional[AgentRepository] = None):
        """
        Initialize the evaluator.
        
        Args:
            repository: Optional agent repository for storing results
        """
        self.rubric = EvaluationRubric()
        self.repository = repository
    
    def evaluate_agent(self, agent_id: str, agent_name: str, 
                       problem: str, discussion: List[Dict], 
                       solution: str) -> Dict[str, Any]:
        """
        Evaluate an agent's performance and record results.
        
        Args:
            agent_id: Agent's unique ID
            agent_name: Agent's name
            problem: Original problem statement
            discussion: Full discussion history
            solution: Final solution
            
        Returns:
            Evaluation results
        """
        # Extract agent's contributions
        agent_contributions = [
            entry["content"] for entry in discussion 
            if entry.get("sender") == agent_name
        ]
        
        # Skip evaluation if agent didn't contribute
        if not agent_contributions:
            return {
                "agent_id": agent_id,
                "evaluation_timestamp": datetime.now().isoformat(),
                "metrics": {},
                "explanations": ["Agent did not contribute to the discussion"]
            }
        
        # Combine contributions for evaluation
        combined_contribution = " ".join(agent_contributions)
        
        # Evaluate context preservation
        context_score, context_explanations = self.rubric.evaluate_context_preservation(
            problem, combined_contribution
        )
        
        # Evaluate reference quality
        reference_score, reference_explanations = self.rubric.evaluate_reference_quality(
            discussion, agent_name, combined_contribution
        )
        
        # Evaluate pragmatism
        pragmatism_score, pragmatism_explanations = self.rubric.evaluate_pragmatism(
            combined_contribution
        )
        
        # Evaluate creativity (simplified approach)
        creativity_score = 0.6  # Default score
        
        # Compile metrics
        metrics = {
            "context_preservation": context_score,
            "reference_quality": reference_score,
            "pragmatism_score": pragmatism_score,
            "creativity_score": creativity_score,
            "tokens_consumed": len(combined_contribution.split())
        }
        
        # Record in repository if available
        if self.repository:
            self.repository.record_interaction(agent_id, metrics)
        
        # Return evaluation results
        return {
            "agent_id": agent_id,
            "evaluation_timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "explanations": {
                "context": context_explanations,
                "references": reference_explanations,
                "pragmatism": pragmatism_explanations
            }
        }
    
    def evaluate_team(self, problem: str, discussion: List[Dict], solution: str) -> Dict[str, Any]:
        """
        Evaluate the performance of the entire team.
        
        Args:
            problem: Original problem statement
            discussion: Full discussion history
            solution: Final solution
            
        Returns:
            Team evaluation results
        """
        # Evaluate solution context preservation
        context_score, context_explanations = self.rubric.evaluate_context_preservation(
            problem, solution
        )
        
        # Evaluate solution pragmatism
        pragmatism_score, pragmatism_explanations = self.rubric.evaluate_pragmatism(
            solution
        )
        
        # Evaluate team collaboration (simple approach)
        agent_interactions = {}
        for entry in discussion:
            sender = entry.get("sender")
            if sender:
                if sender not in agent_interactions:
                    agent_interactions[sender] = {"messages": 0, "references": 0}
                agent_interactions[sender]["messages"] += 1
                
                # Count references to other agents
                for other_agent in agent_interactions.keys():
                    if other_agent != sender and other_agent in entry["content"]:
                        agent_interactions[sender]["references"] += 1
        
        # Calculate collaboration score
        collaboration_score = 0.5  # Default score
        if agent_interactions:
            total_references = sum(agent["references"] for agent in agent_interactions.values())
            total_messages = sum(agent["messages"] for agent in agent_interactions.values())
            if total_messages > 1:
                reference_ratio = total_references / (total_messages - 1)
                collaboration_score = min(1.0, 0.4 + (reference_ratio * 0.6))
        
        # Return team evaluation
        return {
            "evaluation_timestamp": datetime.now().isoformat(),
            "metrics": {
                "context_preservation": context_score,
                "pragmatism_score": pragmatism_score,
                "collaboration_score": collaboration_score,
                "overall_score": (context_score + pragmatism_score + collaboration_score) / 3
            },
            "explanations": {
                "context": context_explanations,
                "pragmatism": pragmatism_explanations,
                "collaboration": [
                    f"Total agent interactions: {len(agent_interactions)}",
                    f"Reference ratio: {total_references}/{total_messages}" if 'total_messages' in locals() else "No messages"
                ]
            }
        }
