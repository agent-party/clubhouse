"""
Evolution Proposal Service Implementation.

This module provides the EvolutionProposalService which is responsible for
generating evolution proposals based on observations and storing them.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4

from agent_orchestration.core.service_base import ServiceBase
from agent_orchestration.evolution.models import (
    EvolutionProposal, 
    EvolutionTrigger, 
    Observation,
    ObservationSource,
    Change,
    ExpectedOutcome,
    ImpactLevel,
    ComplexityLevel,
    ProposalStatus,
    CapabilityChange,
    ServiceChange,
    ModelChange
)
from agent_orchestration.infrastructure.errors import (
    ValidationError, 
    ResourceNotFoundError,
    DuplicateResourceError
)
from agent_orchestration.integration.event_publisher import EventPublisher

logger = logging.getLogger(__name__)


class ObservationService(ServiceBase):
    """Service for managing observations that may trigger evolution proposals."""
    
    def __init__(self, event_publisher: EventPublisher):
        """
        Initialize ObservationService.
        
        Args:
            event_publisher: Event publisher for emitting events
        """
        self.event_publisher = event_publisher
        self.observations: Dict[str, Observation] = {}
    
    async def add_observation(self, observation: Union[Observation, Dict[str, Any]]) -> str:
        """
        Add a new observation.
        
        Args:
            observation: Observation to add, either as a model or dictionary
            
        Returns:
            ID of the added observation
            
        Raises:
            ValidationError: If observation is invalid
        """
        try:
            # Convert to model if dictionary
            if isinstance(observation, dict):
                observation = Observation(**observation)
            
            # Store observation
            self.observations[observation.id] = observation
            
            # Publish event
            await self.event_publisher.publish(
                topic="evolution.observations",
                key=observation.id,
                value={
                    "type": "observation_added",
                    "observation_id": observation.id,
                    "source": observation.source,
                    "target_system": observation.target_system,
                    "category": observation.category,
                    "importance": observation.importance,
                    "timestamp": int(time.time() * 1000)
                }
            )
            
            logger.info(f"Added observation {observation.id} for {observation.target_system}")
            return observation.id
            
        except Exception as e:
            logger.error(f"Error adding observation: {str(e)}", exc_info=e)
            raise ValidationError(f"Invalid observation: {str(e)}")
    
    async def get_observation(self, observation_id: str) -> Observation:
        """
        Get an observation by ID.
        
        Args:
            observation_id: ID of the observation
            
        Returns:
            Observation
            
        Raises:
            ResourceNotFoundError: If observation not found
        """
        if observation_id not in self.observations:
            raise ResourceNotFoundError(f"Observation {observation_id} not found")
        
        return self.observations[observation_id]
    
    async def get_observations_for_system(
        self, target_system: str, limit: int = 100, importance_threshold: float = 0.0
    ) -> List[Observation]:
        """
        Get observations for a specific system.
        
        Args:
            target_system: Target system to get observations for
            limit: Maximum number of observations to return
            importance_threshold: Minimum importance score
            
        Returns:
            List of observations for the system
        """
        # Filter observations for the target system with importance above threshold
        filtered = [
            obs for obs in self.observations.values()
            if obs.target_system == target_system and obs.importance >= importance_threshold
        ]
        
        # Sort by importance (descending) and timestamp (descending)
        sorted_observations = sorted(
            filtered,
            key=lambda obs: (-obs.importance, -obs.timestamp.timestamp())
        )
        
        return sorted_observations[:limit]


class EvolutionProposalService(ServiceBase):
    """Service for generating and managing evolution proposals."""
    
    def __init__(
        self,
        observation_service: ObservationService,
        event_publisher: EventPublisher,
        mcp_service: Optional[Any] = None,
    ):
        """
        Initialize EvolutionProposalService.
        
        Args:
            observation_service: Observation service for accessing observations
            event_publisher: Event publisher for emitting events
            mcp_service: Optional MCP service for generating proposals
        """
        self.observation_service = observation_service
        self.event_publisher = event_publisher
        self.mcp_service = mcp_service
        self.proposals: Dict[str, EvolutionProposal] = {}
    
    async def generate_proposal(
        self,
        target_system: str,
        agent_id: str,
        trigger_observation_id: Optional[str] = None,
        use_mcp: bool = True
    ) -> EvolutionProposal:
        """
        Generate an evolution proposal for a target system.
        
        Args:
            target_system: Target system for the evolution
            agent_id: ID of the agent generating the proposal
            trigger_observation_id: ID of the observation that triggered the proposal
            use_mcp: Whether to use MCP for generating the proposal
            
        Returns:
            Generated evolution proposal
            
        Raises:
            ValidationError: If proposal generation fails
            ResourceNotFoundError: If trigger observation not found
        """
        try:
            # Collect relevant observations
            observations = await self.observation_service.get_observations_for_system(
                target_system=target_system,
                importance_threshold=0.3
            )
            
            if not observations:
                raise ValidationError(f"No relevant observations found for {target_system}")
            
            # Get trigger observation
            trigger_observation = None
            if trigger_observation_id:
                trigger_observation = await self.observation_service.get_observation(
                    observation_id=trigger_observation_id
                )
            else:
                # Use most important observation as trigger
                trigger_observation = max(observations, key=lambda obs: obs.importance)
            
            # Create trigger from observation
            trigger = EvolutionTrigger(
                type=self._map_observation_to_trigger_type(trigger_observation.source),
                source=trigger_observation.source,
                description=trigger_observation.description,
                metrics=trigger_observation.data,
                timestamp=trigger_observation.timestamp
            )
            
            # Generate proposal content
            if use_mcp and self.mcp_service:
                # Use MCP to generate proposal
                proposal_content = await self._generate_proposal_with_mcp(
                    target_system=target_system,
                    trigger=trigger,
                    observations=observations
                )
            else:
                # Generate basic proposal without MCP
                proposal_content = await self._generate_basic_proposal(
                    target_system=target_system,
                    trigger=trigger,
                    observations=observations
                )
            
            # Create full proposal
            proposal = EvolutionProposal(
                title=proposal_content["title"],
                description=proposal_content["description"],
                trigger=trigger,
                target_system=target_system,
                changes=proposal_content["changes"],
                expected_outcomes=proposal_content["expected_outcomes"],
                impact_level=proposal_content["impact_level"],
                complexity=proposal_content["complexity"],
                agent_id=agent_id,
                status=ProposalStatus.DRAFT
            )
            
            # Store proposal
            self.proposals[proposal.id] = proposal
            
            # Publish event
            await self.event_publisher.publish(
                topic="evolution.proposals",
                key=proposal.id,
                value={
                    "type": "evolution_proposal_created",
                    "proposal_id": proposal.id,
                    "agent_id": agent_id,
                    "target_system": target_system,
                    "impact_level": proposal.impact_level,
                    "timestamp": int(time.time() * 1000)
                }
            )
            
            logger.info(f"Generated evolution proposal {proposal.id} for {target_system}")
            return proposal
            
        except ResourceNotFoundError:
            logger.error(f"Trigger observation {trigger_observation_id} not found")
            raise
            
        except Exception as e:
            logger.error(f"Error generating proposal: {str(e)}", exc_info=e)
            raise ValidationError(f"Failed to generate evolution proposal: {str(e)}")
    
    async def submit_proposal(self, proposal_id: str) -> EvolutionProposal:
        """
        Submit an evolution proposal for validation.
        
        Args:
            proposal_id: ID of the proposal to submit
            
        Returns:
            Updated evolution proposal
            
        Raises:
            ResourceNotFoundError: If proposal not found
            ValidationError: If proposal cannot be submitted
        """
        if proposal_id not in self.proposals:
            raise ResourceNotFoundError(f"Proposal {proposal_id} not found")
        
        proposal = self.proposals[proposal_id]
        
        if proposal.status != ProposalStatus.DRAFT:
            raise ValidationError(f"Cannot submit proposal with status {proposal.status}")
        
        # Update status
        proposal.status = ProposalStatus.SUBMITTED
        proposal.updated_at = datetime.now()
        
        # Publish event
        await self.event_publisher.publish(
            topic="evolution.proposals",
            key=proposal.id,
            value={
                "type": "evolution_proposal_submitted",
                "proposal_id": proposal.id,
                "agent_id": proposal.agent_id,
                "target_system": proposal.target_system,
                "timestamp": int(time.time() * 1000)
            }
        )
        
        logger.info(f"Submitted evolution proposal {proposal.id}")
        return proposal
    
    async def get_proposal(self, proposal_id: str) -> EvolutionProposal:
        """
        Get an evolution proposal by ID.
        
        Args:
            proposal_id: ID of the proposal
            
        Returns:
            Evolution proposal
            
        Raises:
            ResourceNotFoundError: If proposal not found
        """
        if proposal_id not in self.proposals:
            raise ResourceNotFoundError(f"Proposal {proposal_id} not found")
        
        return self.proposals[proposal_id]
    
    async def get_proposals_for_system(
        self, target_system: str, status: Optional[ProposalStatus] = None
    ) -> List[EvolutionProposal]:
        """
        Get evolution proposals for a specific system.
        
        Args:
            target_system: Target system to get proposals for
            status: Optional status filter
            
        Returns:
            List of evolution proposals for the system
        """
        # Filter proposals for the target system and optional status
        filtered = [
            prop for prop in self.proposals.values()
            if prop.target_system == target_system and
            (status is None or prop.status == status)
        ]
        
        # Sort by created_at (descending)
        return sorted(filtered, key=lambda prop: -prop.created_at.timestamp())
    
    async def update_proposal_status(
        self, proposal_id: str, status: ProposalStatus, reason: Optional[str] = None
    ) -> EvolutionProposal:
        """
        Update the status of an evolution proposal.
        
        Args:
            proposal_id: ID of the proposal
            status: New status
            reason: Optional reason for the status change
            
        Returns:
            Updated evolution proposal
            
        Raises:
            ResourceNotFoundError: If proposal not found
        """
        if proposal_id not in self.proposals:
            raise ResourceNotFoundError(f"Proposal {proposal_id} not found")
        
        proposal = self.proposals[proposal_id]
        old_status = proposal.status
        
        # Update status
        proposal.status = status
        proposal.updated_at = datetime.now()
        
        # Add reason to metadata if provided
        if reason:
            proposal.metadata["status_change_reason"] = reason
        
        # Publish event
        await self.event_publisher.publish(
            topic="evolution.proposals",
            key=proposal.id,
            value={
                "type": "evolution_proposal_status_changed",
                "proposal_id": proposal.id,
                "old_status": old_status,
                "new_status": status,
                "reason": reason,
                "timestamp": int(time.time() * 1000)
            }
        )
        
        logger.info(f"Updated proposal {proposal.id} status to {status}")
        return proposal
    
    def _map_observation_to_trigger_type(self, source: Union[str, ObservationSource]) -> str:
        """
        Map observation source to trigger type.
        
        Args:
            source: Observation source
            
        Returns:
            Trigger type
        """
        source_str = source
        if isinstance(source, ObservationSource):
            source_str = source.value
            
        mapping = {
            "user_feedback": "user_feedback",
            "agent_feedback": "agent_feedback",
            "performance_metric": "performance",
            "error_log": "error",
            "usage_pattern": "usage"
        }
        
        return mapping.get(source_str, "other")
    
    async def _generate_proposal_with_mcp(
        self, target_system: str, trigger: EvolutionTrigger, observations: List[Observation]
    ) -> Dict[str, Any]:
        """
        Generate proposal content using MCP service.
        
        Args:
            target_system: Target system for the evolution
            trigger: Trigger for the proposal
            observations: Relevant observations
            
        Returns:
            Proposal content
        """
        if not self.mcp_service:
            raise ValueError("MCP service not available")
        
        # Prepare context for MCP
        context = {
            "target_system": target_system,
            "trigger": trigger.dict(),
            "observations": [obs.dict() for obs in observations]
        }
        
        # Call MCP to generate proposal
        try:
            response = await self.mcp_service.generate_evolution_proposal(context=context)
            
            # Extract and validate proposal content
            proposal_content = response["proposal"]
            
            # Validate required fields
            required_fields = ["title", "description", "changes", "expected_outcomes", 
                               "impact_level", "complexity"]
            for field in required_fields:
                if field not in proposal_content:
                    raise ValidationError(f"MCP response missing required field: {field}")
            
            # Convert changes and expected outcomes to proper models
            proposal_content["changes"] = self._convert_changes(proposal_content["changes"])
            proposal_content["expected_outcomes"] = self._convert_outcomes(
                proposal_content["expected_outcomes"]
            )
            
            return proposal_content
            
        except Exception as e:
            logger.error(f"Error generating proposal with MCP: {str(e)}", exc_info=e)
            raise ValidationError(f"Failed to generate proposal with MCP: {str(e)}")
    
    async def _generate_basic_proposal(
        self, target_system: str, trigger: EvolutionTrigger, observations: List[Observation]
    ) -> Dict[str, Any]:
        """
        Generate basic proposal content without MCP.
        
        This is a fallback method when MCP is not available or not used.
        
        Args:
            target_system: Target system for the evolution
            trigger: Trigger for the proposal
            observations: Relevant observations
            
        Returns:
            Proposal content
        """
        # Determine most common category from observations
        categories = [obs.category for obs in observations]
        most_common_category = max(set(categories), key=categories.count)
        
        # Calculate average importance
        avg_importance = sum(obs.importance for obs in observations) / len(observations)
        
        # Determine impact level based on importance
        impact_level = ImpactLevel.LOW
        if avg_importance > 0.7:
            impact_level = ImpactLevel.HIGH
        elif avg_importance > 0.4:
            impact_level = ImpactLevel.MEDIUM
        
        # Create basic changes
        changes = []
        if most_common_category == "performance":
            changes.append(Change(
                type="service",
                description=f"Optimize {target_system} performance",
                details=ServiceChange(
                    service_name=target_system,
                    change_type="modify",
                    description=f"Optimize {target_system} for better performance",
                    implementation_changes={
                        "focus_areas": ["caching", "query_optimization"]
                    }
                ),
                rationale="Multiple performance observations indicate optimization is needed",
                impact_assessment="Should improve response times and resource utilization"
            ))
        elif most_common_category == "error":
            changes.append(Change(
                type="service",
                description=f"Fix errors in {target_system}",
                details=ServiceChange(
                    service_name=target_system,
                    change_type="modify",
                    description=f"Fix error handling in {target_system}",
                    implementation_changes={
                        "focus_areas": ["error_handling", "validation"]
                    }
                ),
                rationale="Error observations indicate issues that need resolution",
                impact_assessment="Should improve reliability and user experience"
            ))
        else:
            changes.append(Change(
                type="capability",
                description=f"Enhance {target_system} capabilities",
                details=CapabilityChange(
                    capability_name=f"{target_system}_capability",
                    change_type="modify",
                    description=f"Enhance capabilities based on observations",
                    parameters={
                        "improvements": ["usability", "functionality"]
                    }
                ),
                rationale="Observations indicate room for improvement in capabilities",
                impact_assessment="Should enhance user experience and system utility"
            ))
        
        # Create expected outcomes
        expected_outcomes = [
            ExpectedOutcome(
                description=f"Improved {target_system} based on observations",
                metrics={
                    "performance": "response_time",
                    "reliability": "error_rate",
                    "satisfaction": "user_rating"
                },
                validation_criteria=[
                    f"Reduced number of observations in {most_common_category} category",
                    "Positive user feedback",
                    "Improved system metrics"
                ]
            )
        ]
        
        return {
            "title": f"{target_system.capitalize()} Evolution: {most_common_category.capitalize()} Improvement",
            "description": f"Evolution proposal to address {most_common_category} issues in {target_system} based on {len(observations)} observations",
            "changes": changes,
            "expected_outcomes": expected_outcomes,
            "impact_level": impact_level,
            "complexity": ComplexityLevel.MODERATE
        }
    
    def _convert_changes(self, changes_data: List[Dict[str, Any]]) -> List[Change]:
        """
        Convert raw changes data to Change models.
        
        Args:
            changes_data: Raw changes data from MCP
            
        Returns:
            List of Change models
        """
        changes = []
        
        for change_data in changes_data:
            change_type = change_data.get("type")
            details = change_data.get("details", {})
            
            if change_type == "capability":
                details_model = CapabilityChange(**details)
            elif change_type == "service":
                details_model = ServiceChange(**details)
            elif change_type == "model":
                details_model = ModelChange(**details)
            else:
                details_model = details
            
            change = Change(
                type=change_type,
                description=change_data.get("description", ""),
                details=details_model,
                rationale=change_data.get("rationale", ""),
                impact_assessment=change_data.get("impact_assessment", "")
            )
            
            changes.append(change)
        
        return changes
    
    def _convert_outcomes(self, outcomes_data: List[Dict[str, Any]]) -> List[ExpectedOutcome]:
        """
        Convert raw outcomes data to ExpectedOutcome models.
        
        Args:
            outcomes_data: Raw outcomes data from MCP
            
        Returns:
            List of ExpectedOutcome models
        """
        return [ExpectedOutcome(**outcome) for outcome in outcomes_data]
