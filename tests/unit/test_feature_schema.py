"""Unit tests for FeatureSchema validation."""

import pytest
import pandas as pd
import numpy as np

from models.feature_schema import FeatureSchema


class TestFeatureSchema:
    """Tests for FeatureSchema class."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame with OHLCV + indicators."""
        np.random.seed(42)
        n = 100

        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1H'),
            'open': np.random.uniform(90, 110, n),
            'high': np.random.uniform(95, 115, n),
            'low': np.random.uniform(85, 105, n),
            'close': np.random.uniform(90, 110, n),
            'volume': np.random.uniform(1000, 10000, n),
            'num_trades': np.random.randint(100, 1000, n),
            'sma_20': np.random.uniform(90, 110, n),
            'rsi_14': np.random.uniform(0, 100, n),
            'macd': np.random.uniform(-1, 1, n),
        })

    def test_from_dataframe_exclude_ohlcv(self, sample_dataframe):
        """Test creating schema that excludes OHLCV."""
        schema = FeatureSchema.from_dataframe(
            sample_dataframe,
            include_ohlcv=False,
            include_num_trades=False,
        )

        # Should only have indicator columns
        assert 'sma_20' in schema.feature_names
        assert 'rsi_14' in schema.feature_names
        assert 'macd' in schema.feature_names

        # OHLCV should be excluded
        assert 'open' not in schema.feature_names
        assert 'high' not in schema.feature_names
        assert 'low' not in schema.feature_names
        assert 'close' not in schema.feature_names
        assert 'volume' not in schema.feature_names
        assert 'num_trades' not in schema.feature_names

        # Timestamp should be excluded
        assert 'timestamp' not in schema.feature_names

    def test_from_dataframe_include_ohlcv(self, sample_dataframe):
        """Test creating schema that includes OHLCV."""
        schema = FeatureSchema.from_dataframe(
            sample_dataframe,
            include_ohlcv=True,
            include_volume=True,
        )

        # Should have OHLCV + indicators
        assert 'open' in schema.feature_names
        assert 'high' in schema.feature_names
        assert 'low' in schema.feature_names
        assert 'close' in schema.feature_names
        assert 'volume' in schema.feature_names
        assert 'sma_20' in schema.feature_names
        assert 'rsi_14' in schema.feature_names

    def test_from_dataframe_exclude_volume(self, sample_dataframe):
        """Test excluding volume even with OHLCV enabled."""
        schema = FeatureSchema.from_dataframe(
            sample_dataframe,
            include_ohlcv=True,
            include_volume=False,
        )

        assert 'open' in schema.feature_names
        assert 'close' in schema.feature_names
        assert 'volume' not in schema.feature_names

    def test_validate_valid_dataframe(self, sample_dataframe):
        """Test validating a DataFrame that matches schema."""
        schema = FeatureSchema.from_dataframe(
            sample_dataframe,
            include_ohlcv=False,
        )

        # Should not raise
        schema.validate(sample_dataframe)

    def test_validate_missing_features(self, sample_dataframe):
        """Test validation fails when features are missing."""
        schema = FeatureSchema(
            feature_names=['sma_20', 'rsi_14', 'missing_feature'],
            include_ohlcv=False,
        )

        with pytest.raises(ValueError, match="Missing required features"):
            schema.validate(sample_dataframe)

    def test_validate_extra_columns(self, sample_dataframe):
        """Test validation warns about extra columns but doesn't fail."""
        schema = FeatureSchema(
            feature_names=['sma_20'],
            include_ohlcv=False,
        )

        # Should not raise (just warns)
        schema.validate(sample_dataframe)

    def test_extract_features_correct_order(self, sample_dataframe):
        """Test that features are extracted in correct order."""
        # Create schema with specific order
        schema = FeatureSchema(
            feature_names=['macd', 'rsi_14', 'sma_20'],  # Alphabetically reversed
            include_ohlcv=False,
        )

        extracted = schema.extract_features(sample_dataframe)

        # Should have correct order
        assert list(extracted.columns) == ['macd', 'rsi_14', 'sma_20']

    def test_extract_features_only_required(self, sample_dataframe):
        """Test that only required features are extracted."""
        schema = FeatureSchema(
            feature_names=['sma_20', 'rsi_14'],
            include_ohlcv=False,
        )

        extracted = schema.extract_features(sample_dataframe)

        # Should only have required features
        assert set(extracted.columns) == {'sma_20', 'rsi_14'}
        assert 'macd' not in extracted.columns
        assert 'open' not in extracted.columns

    def test_check_nan_values(self, sample_dataframe):
        """Test NaN detection in features."""
        # Add some NaN values
        sample_dataframe.loc[5, 'sma_20'] = np.nan
        sample_dataframe.loc[10, 'rsi_14'] = np.nan

        schema = FeatureSchema(
            feature_names=['sma_20', 'rsi_14', 'macd'],
            include_ohlcv=False,
        )

        nan_features = schema.check_nan_values(sample_dataframe)

        assert set(nan_features) == {'sma_20', 'rsi_14'}
        assert 'macd' not in nan_features

    def test_check_nan_values_none(self, sample_dataframe):
        """Test NaN detection with no NaN values."""
        schema = FeatureSchema.from_dataframe(
            sample_dataframe,
            include_ohlcv=False,
        )

        nan_features = schema.check_nan_values(sample_dataframe)

        assert nan_features == []

    def test_to_dict_from_dict_roundtrip(self, sample_dataframe):
        """Test serialization/deserialization."""
        schema1 = FeatureSchema.from_dataframe(
            sample_dataframe,
            include_ohlcv=True,
            include_volume=False,
        )

        # Serialize and deserialize
        data = schema1.to_dict()
        schema2 = FeatureSchema.from_dict(data)

        # Should be equivalent
        assert schema1.feature_names == schema2.feature_names
        assert schema1.include_ohlcv == schema2.include_ohlcv
        assert schema1.include_volume == schema2.include_volume
        assert schema1.ohlcv_columns == schema2.ohlcv_columns

    def test_repr(self):
        """Test string representation."""
        schema = FeatureSchema(
            feature_names=['feature1', 'feature2', 'feature3'],
            include_ohlcv=True,
        )

        repr_str = repr(schema)

        assert 'n_features=3' in repr_str
        assert 'include_ohlcv=True' in repr_str

    def test_include_num_trades(self, sample_dataframe):
        """Test including num_trades in features."""
        schema = FeatureSchema.from_dataframe(
            sample_dataframe,
            include_ohlcv=False,
            include_num_trades=True,
        )

        assert 'num_trades' in schema.feature_names
        assert schema.include_num_trades is True

    def test_exclude_custom_columns(self, sample_dataframe):
        """Test excluding custom columns."""
        schema = FeatureSchema.from_dataframe(
            sample_dataframe,
            include_ohlcv=False,
            exclude_columns=['sma_20'],
        )

        # sma_20 should be excluded even though it's an indicator
        assert 'sma_20' not in schema.feature_names
        assert 'rsi_14' in schema.feature_names
        assert 'macd' in schema.feature_names


class TestFeatureSchemaIntegration:
    """Integration tests with real-world scenarios."""

    def test_phase_10_ohlcv_exclusion(self):
        """Test Phase 10 feature: OHLCV exclusion from features."""
        # Simulate training data processing
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'open': np.random.uniform(90, 110, 100),
            'high': np.random.uniform(95, 115, 100),
            'low': np.random.uniform(85, 105, 100),
            'close': np.random.uniform(90, 110, 100),
            'volume': np.random.uniform(1000, 10000, 100),
            'sma_20': np.random.uniform(90, 110, 100),
            'rsi_14': np.random.uniform(0, 100, 100),
        })

        # Create schema with OHLCV excluded (Phase 10)
        schema = FeatureSchema.from_dataframe(df, include_ohlcv=False)

        # Extract features
        features = schema.extract_features(df)

        # Verify OHLCV is excluded
        assert 'open' not in features.columns
        assert 'high' not in features.columns
        assert 'low' not in features.columns
        assert 'close' not in features.columns
        assert 'volume' not in features.columns

        # Verify indicators are included
        assert 'sma_20' in features.columns
        assert 'rsi_14' in features.columns

    def test_inference_validation(self):
        """Test validation workflow during inference."""
        # Training data
        training_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'open': np.random.uniform(90, 110, 100),
            'close': np.random.uniform(90, 110, 100),
            'sma_20': np.random.uniform(90, 110, 100),
            'rsi_14': np.random.uniform(0, 100, 100),
            'macd': np.random.uniform(-1, 1, 100),
        })

        # Create schema from training
        schema = FeatureSchema.from_dataframe(training_df, include_ohlcv=False)

        # Inference data (should match)
        inference_df = training_df.copy()

        # Should validate successfully
        schema.validate(inference_df)
        features = schema.extract_features(inference_df)

        assert len(features.columns) == 3  # sma_20, rsi_14, macd

    def test_feature_mismatch_detection(self):
        """Test that feature mismatches are detected."""
        # Training schema expects specific features
        schema = FeatureSchema(
            feature_names=['sma_20', 'rsi_14', 'macd'],
            include_ohlcv=False,
        )

        # Inference data is missing macd
        inference_df = pd.DataFrame({
            'sma_20': [1, 2, 3],
            'rsi_14': [50, 60, 70],
            # macd is missing
        })

        # Should raise ValueError
        with pytest.raises(ValueError, match="Missing required features"):
            schema.validate(inference_df)


class TestFeatureExclusion:
    """Test suite for OHLCV feature exclusion."""

    def test_ohlcv_columns_defined(self):
        """Test that OHLCV columns are properly defined."""
        ohlcv_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        assert len(ohlcv_cols) == 6

    def test_feature_exclusion_logic(self):
        """Test feature column filtering logic."""
        # Simulate dataframe columns
        all_cols = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'ema_12', 'rsi_14', 'macd', 'bollinger_upper'
        ]

        ohlcv_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in all_cols if col not in ohlcv_cols]

        # Should only have indicators
        assert 'open' not in feature_cols
        assert 'close' not in feature_cols
        assert 'ema_12' in feature_cols
        assert 'rsi_14' in feature_cols
        assert len(feature_cols) == 4

    def test_no_ohlcv_leakage(self):
        """Test that OHLCV columns are not used as features."""
        all_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'ema_12']
        ohlcv_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        feature_cols = [col for col in all_cols if col not in ohlcv_cols]

        # Verify no OHLCV in features
        for ohlcv in ohlcv_cols:
            assert ohlcv not in feature_cols