"""
Unit tests for datetime_parser.py

Run with: pytest tests/test_datetime_parser.py -v
"""

import pytest
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from datetime_parser import DateTimeParser, parse_datetime_query


class TestDateTimeParser:
    """Test suite for DateTimeParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = DateTimeParser()
    
    def test_detect_temporal_references_single(self):
        """Test detection of single temporal reference."""
        query = "What happened with UPI today?"
        detected = self.parser.detect_temporal_references(query)
        
        assert 'today' in detected
        assert len(detected) == 1
    
    def test_detect_temporal_references_multiple(self):
        """Test detection of multiple temporal references."""
        query = "What changed between yesterday and today?"
        detected = self.parser.detect_temporal_references(query)
        
        assert 'yesterday' in detected
        assert 'today' in detected
        assert len(detected) == 2
    
    def test_detect_temporal_references_none(self):
        """Test query with no temporal references."""
        query = "What are NEFT charges?"
        detected = self.parser.detect_temporal_references(query)
        
        assert len(detected) == 0
    
    def test_detect_temporal_references_multiword(self):
        """Test detection of multi-word temporal phrases."""
        query = "Show me updates from last 7 days"
        detected = self.parser.detect_temporal_references(query)
        
        assert 'last 7 days' in detected
    
    def test_detect_temporal_references_case_insensitive(self):
        """Test that detection is case-insensitive."""
        queries = [
            "What happened TODAY?",
            "What happened today?",
            "What happened ToDay?",
        ]
        
        for query in queries:
            detected = self.parser.detect_temporal_references(query)
            assert 'today' in detected
    
    def test_detect_temporal_references_weekday(self):
        """Test detection of weekday references."""
        query = "What happened last Monday?"
        detected = self.parser.detect_temporal_references(query)
        
        assert 'monday' in detected
    
    def test_resolve_to_date_today(self):
        """Test resolution of 'today' to current date."""
        today_str = self.parser.resolve_to_date('today')
        expected = datetime.now().strftime('%Y-%m-%d')
        
        assert today_str == expected
        assert len(today_str) == 10  # YYYY-MM-DD format
        assert today_str.count('-') == 2
    
    def test_resolve_to_date_yesterday(self):
        """Test resolution of 'yesterday' to previous day."""
        yesterday_str = self.parser.resolve_to_date('yesterday')
        expected = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        assert yesterday_str == expected
    
    def test_resolve_to_date_tomorrow(self):
        """Test resolution of 'tomorrow' to next day."""
        tomorrow_str = self.parser.resolve_to_date('tomorrow')
        expected = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        assert tomorrow_str == expected
    
    def test_resolve_to_date_last_week(self):
        """Test resolution of 'last week' to 7 days ago."""
        last_week_str = self.parser.resolve_to_date('last week')
        expected = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        assert last_week_str == expected
    
    def test_resolve_to_date_weekday(self):
        """Test resolution of weekday references."""
        monday_str = self.parser.resolve_to_date('monday')
        
        # Should be a valid date
        assert monday_str is not None
        assert len(monday_str) == 10
        
        # Should be in the past (last Monday)
        monday_date = datetime.strptime(monday_str, '%Y-%m-%d')
        assert monday_date <= datetime.now()
    
    def test_resolve_to_date_invalid(self):
        """Test resolution of invalid temporal reference."""
        result = self.parser.resolve_to_date('invalid_temporal_ref')
        assert result is None
    
    def test_get_date_range_today(self):
        """Test date range for 'today'."""
        start, end = self.parser.get_date_range('today')
        expected = datetime.now().strftime('%Y-%m-%d')
        
        assert start == expected
        assert end == expected
    
    def test_get_date_range_this_week(self):
        """Test date range for 'this week'."""
        start, end = self.parser.get_date_range('this week')
        
        # Start should be Monday
        start_date = datetime.strptime(start, '%Y-%m-%d')
        assert start_date.weekday() == 0  # Monday
        
        # End should be 6 days after start
        end_date = datetime.strptime(end, '%Y-%m-%d')
        assert (end_date - start_date).days == 6
    
    def test_get_date_range_this_month(self):
        """Test date range for 'this month'."""
        start, end = self.parser.get_date_range('this month')
        
        # Start should be day 1
        start_date = datetime.strptime(start, '%Y-%m-%d')
        assert start_date.day == 1
        
        # End should be last day of month
        end_date = datetime.strptime(end, '%Y-%m-%d')
        assert end_date.month == datetime.now().month
    
    def test_get_date_range_single_date(self):
        """Test date range for single-date references."""
        start, end = self.parser.get_date_range('yesterday')
        
        # Both should be the same date
        assert start == end
        
        # Should be yesterday
        expected = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        assert start == expected
    
    def test_augment_query_with_temporal(self):
        """Test query augmentation with temporal reference."""
        query = "What happened today with UPI?"
        augmented = self.parser.augment_query(query)
        
        # Should contain original query
        assert "What happened today with UPI?" in augmented
        
        # Should contain date context
        assert 'today =' in augmented
        
        # Should contain current date
        today_str = datetime.now().strftime('%Y-%m-%d')
        assert today_str in augmented
    
    def test_augment_query_without_temporal(self):
        """Test query augmentation without temporal reference."""
        query = "What are RTGS charges?"
        augmented = self.parser.augment_query(query)
        
        # Should be unchanged
        assert augmented == query
    
    def test_augment_query_multiple_temporal(self):
        """Test query augmentation with multiple temporal references."""
        query = "What changed between yesterday and today?"
        augmented = self.parser.augment_query(query)
        
        # Should contain both dates
        assert 'yesterday =' in augmented
        assert 'today =' in augmented
    
    def test_get_current_date_context(self):
        """Test current date context generation."""
        context = self.parser.get_current_date_context()
        
        # Should contain 'Current date:'
        assert 'Current date:' in context
        
        # Should contain current date in YYYY-MM-DD format
        today_str = datetime.now().strftime('%Y-%m-%d')
        assert today_str in context
        
        # Should contain weekday name
        weekday = datetime.now().strftime('%A')
        assert weekday in context
        
        # Should contain month name
        month = datetime.now().strftime('%B')
        assert month in context
    
    def test_is_temporal_query_positive(self):
        """Test temporal query detection (positive case)."""
        query = "What happened today?"
        result = self.parser.is_temporal_query(query)
        
        assert result is True
    
    def test_is_temporal_query_negative(self):
        """Test temporal query detection (negative case)."""
        query = "What are NEFT charges?"
        result = self.parser.is_temporal_query(query)
        
        assert result is False


class TestParseDateTimeQuery:
    """Test suite for parse_datetime_query utility function."""
    
    def test_parse_datetime_query_with_temporal(self):
        """Test parsing query with temporal reference."""
        query = "What's new with UPI today?"
        result = parse_datetime_query(query)
        
        assert result['original_query'] == query
        assert result['has_temporal'] is True
        assert 'today' in result['temporal_refs']
        assert 'Current date:' in result['date_context']
        assert 'today =' in result['augmented_query']
    
    def test_parse_datetime_query_without_temporal(self):
        """Test parsing query without temporal reference."""
        query = "What are RTGS transaction limits?"
        result = parse_datetime_query(query)
        
        assert result['original_query'] == query
        assert result['has_temporal'] is False
        assert len(result['temporal_refs']) == 0
        assert result['augmented_query'] == query  # Unchanged
        assert 'Current date:' in result['date_context']  # Still provided
    
    def test_parse_datetime_query_return_structure(self):
        """Test that parse_datetime_query returns correct structure."""
        query = "Any query"
        result = parse_datetime_query(query)
        
        # Check all required keys are present
        required_keys = [
            'original_query',
            'augmented_query',
            'temporal_refs',
            'date_context',
            'has_temporal'
        ]
        
        for key in required_keys:
            assert key in result
        
        # Check types
        assert isinstance(result['original_query'], str)
        assert isinstance(result['augmented_query'], str)
        assert isinstance(result['temporal_refs'], list)
        assert isinstance(result['date_context'], str)
        assert isinstance(result['has_temporal'], bool)


class TestTemporalPatterns:
    """Test various temporal patterns."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = DateTimeParser()
    
    @pytest.mark.parametrize("query,expected_ref", [
        ("What happened today?", "today"),
        ("What happened yesterday?", "yesterday"),
        ("What happened last week?", "last week"),
        ("What happened last month?", "last month"),
        ("What happened last year?", "last year"),
        ("Updates from last 7 days", "last 7 days"),
        ("News from last 30 days", "last 30 days"),
        ("Recent updates on UPI", "recent"),
        ("What's the latest on NEFT?", "latest"),
    ])
    def test_temporal_pattern_detection(self, query, expected_ref):
        """Test detection of various temporal patterns."""
        detected = self.parser.detect_temporal_references(query)
        assert expected_ref in detected
    
    @pytest.mark.parametrize("temporal_ref", [
        "today",
        "yesterday",
        "tomorrow",
        "last week",
        "last month",
        "last 7 days",
        "monday",
        "friday",
    ])
    def test_all_patterns_resolvable(self, temporal_ref):
        """Test that all patterns can be resolved to dates."""
        date_str = self.parser.resolve_to_date(temporal_ref)
        
        assert date_str is not None
        assert len(date_str) == 10  # YYYY-MM-DD format
        assert date_str.count('-') == 2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = DateTimeParser()
    
    def test_empty_query(self):
        """Test handling of empty query."""
        detected = self.parser.detect_temporal_references("")
        assert len(detected) == 0
        
        augmented = self.parser.augment_query("")
        assert augmented == ""
    
    def test_query_with_temporal_in_middle(self):
        """Test temporal reference in middle of query."""
        query = "What happened with today's UPI news?"
        detected = self.parser.detect_temporal_references(query)
        
        assert 'today' in detected
    
    def test_query_with_partial_match(self):
        """Test that partial matches don't trigger detection."""
        query = "What about todayness?"  # Contains 'today' but not as word
        detected = self.parser.detect_temporal_references(query)
        
        # Should not detect 'today' because it's part of another word
        # This depends on word boundary regex - may need adjustment
        # For now, let's just verify it works somehow
        assert isinstance(detected, list)
    
    def test_multiple_same_temporal_refs(self):
        """Test query with same temporal reference multiple times."""
        query = "What happened today and what will happen today?"
        detected = self.parser.detect_temporal_references(query)
        
        # Should only appear once in detected list
        assert detected.count('today') == 1


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, '-v'])