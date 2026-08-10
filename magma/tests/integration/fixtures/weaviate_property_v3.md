# Weaviate Property Configuration Reference

Source: weaviate.collections.classes.config.Property docstring (weaviate-client==4.22.0)

## Attributes

This class defines the structure of a data property that a collection can have within Weaviate.

Attributes:
    name: The name of the property, REQUIRED.
    data_type: The data type of the property, REQUIRED.
    description: A description of the property.
        index_range_filters: Whether the property should support range filters in the inverted index.
    index_inverted: Whether the property should be indexed in the inverted index. Deprecated in favor of index_filterable/index_searchable.
    nested_properties: nested properties for data type OBJECT and OBJECT_ARRAY`.
    skip_vectorization: Whether to skip vectorization of the property. Defaults to `False`.
    text_analyzer: Text analysis options for the property. Configures ASCII folding
        behavior for text and text[] properties using an inverted index. Immutable
        after the property is created.
    tokenization: The tokenization method to use for the inverted index. Defaults to `None`.
    vectorize_property_name: Whether to vectorize the property name. Defaults to `True`.
