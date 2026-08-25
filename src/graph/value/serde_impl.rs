use std::fmt;
use std::sync::Arc;

use serde::de::{self, EnumAccess, IntoDeserializer, MapAccess, SeqAccess, VariantAccess, Visitor};
use serde::ser::{
    self, SerializeMap, SerializeSeq, SerializeStruct, SerializeStructVariant, SerializeTuple,
    SerializeTupleStruct, SerializeTupleVariant,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{Repr, Value, ValueError};

pub(super) fn to_value<T: Serialize>(value: T) -> Result<Value, ValueError> {
    value.serialize(ValueSerializer)
}

impl Serialize for Value {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match &self.0 {
            Repr::Null => serializer.serialize_unit(),
            Repr::Bool(value) => serializer.serialize_bool(*value),
            Repr::I64(value) => serializer.serialize_i64(*value),
            Repr::U64(value) => serializer.serialize_u64(*value),
            Repr::F64(value) => serializer.serialize_f64(*value),
            Repr::String(value) => serializer.serialize_str(value),
            Repr::Array(values) => values.serialize(serializer),
            Repr::Object(entries) => {
                let mut map = serializer.serialize_map(Some(entries.len()))?;
                for (key, value) in entries.iter() {
                    map.serialize_entry(key.as_ref(), value)?;
                }
                map.end()
            }
        }
    }
}

struct ValueVisitor;

impl<'de> Visitor<'de> for ValueVisitor {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an OpenAPI-compatible runtime value")
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(Value::NULL)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(Value::NULL)
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(Value::from(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(Value::from(value))
    }

    fn visit_i128<E>(self, value: i128) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        i64::try_from(value)
            .map(Value::from)
            .map_err(|_| E::custom(ValueError::IntegerOutOfRange))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(Value::from(value))
    }

    fn visit_u128<E>(self, value: u128) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        u64::try_from(value)
            .map(Value::from)
            .map_err(|_| E::custom(ValueError::IntegerOutOfRange))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Value::number(value).map_err(E::custom)
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(Value::from(value))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(Value::from(value))
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::with_capacity(sequence.size_hint().unwrap_or(0));
        while let Some(value) = sequence.next_element()? {
            values.push(value);
        }
        Ok(Value::array(values))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut entries = Vec::with_capacity(map.size_hint().unwrap_or(0));
        while let Some((key, value)) = map.next_entry::<String, Value>()? {
            entries.push((key, value));
        }
        Value::object(entries).map_err(<A::Error as de::Error>::custom)
    }
}

impl<'de> Deserialize<'de> for Value {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(ValueVisitor)
    }
}

struct ValueSerializer;

impl Serializer for ValueSerializer {
    type Ok = Value;
    type Error = ValueError;
    type SerializeSeq = SequenceSerializer;
    type SerializeTuple = SequenceSerializer;
    type SerializeTupleStruct = SequenceSerializer;
    type SerializeTupleVariant = TupleVariantSerializer;
    type SerializeMap = ObjectSerializer;
    type SerializeStruct = ObjectSerializer;
    type SerializeStructVariant = StructVariantSerializer;

    fn serialize_bool(self, value: bool) -> Result<Self::Ok, Self::Error> {
        Ok(Value::from(value))
    }

    fn serialize_i8(self, value: i8) -> Result<Self::Ok, Self::Error> {
        Ok(Value::from(value))
    }
    fn serialize_i16(self, value: i16) -> Result<Self::Ok, Self::Error> {
        Ok(Value::from(value))
    }
    fn serialize_i32(self, value: i32) -> Result<Self::Ok, Self::Error> {
        Ok(Value::from(value))
    }
    fn serialize_i64(self, value: i64) -> Result<Self::Ok, Self::Error> {
        Ok(Value::from(value))
    }

    fn serialize_i128(self, value: i128) -> Result<Self::Ok, Self::Error> {
        i64::try_from(value)
            .map(Value::from)
            .map_err(|_| ValueError::IntegerOutOfRange)
    }

    fn serialize_u8(self, value: u8) -> Result<Self::Ok, Self::Error> {
        Ok(Value::from(value))
    }
    fn serialize_u16(self, value: u16) -> Result<Self::Ok, Self::Error> {
        Ok(Value::from(value))
    }
    fn serialize_u32(self, value: u32) -> Result<Self::Ok, Self::Error> {
        Ok(Value::from(value))
    }
    fn serialize_u64(self, value: u64) -> Result<Self::Ok, Self::Error> {
        Ok(Value::from(value))
    }

    fn serialize_u128(self, value: u128) -> Result<Self::Ok, Self::Error> {
        u64::try_from(value)
            .map(Value::from)
            .map_err(|_| ValueError::IntegerOutOfRange)
    }

    fn serialize_f32(self, value: f32) -> Result<Self::Ok, Self::Error> {
        Value::number(f64::from(value))
    }

    fn serialize_f64(self, value: f64) -> Result<Self::Ok, Self::Error> {
        Value::number(value)
    }

    fn serialize_char(self, value: char) -> Result<Self::Ok, Self::Error> {
        Ok(Value::from(value.to_string()))
    }

    fn serialize_str(self, value: &str) -> Result<Self::Ok, Self::Error> {
        Ok(Value::from(value))
    }

    fn serialize_bytes(self, value: &[u8]) -> Result<Self::Ok, Self::Error> {
        Ok(Value::array(value.iter().copied().map(Value::from)))
    }

    fn serialize_none(self) -> Result<Self::Ok, Self::Error> {
        Ok(Value::NULL)
    }

    fn serialize_some<T>(self, value: &T) -> Result<Self::Ok, Self::Error>
    where
        T: ?Sized + Serialize,
    {
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<Self::Ok, Self::Error> {
        Ok(Value::NULL)
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<Self::Ok, Self::Error> {
        Ok(Value::NULL)
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
    ) -> Result<Self::Ok, Self::Error> {
        Ok(Value::from(variant))
    }

    fn serialize_newtype_struct<T>(
        self,
        _name: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Self::Error>
    where
        T: ?Sized + Serialize,
    {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T>(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Self::Error>
    where
        T: ?Sized + Serialize,
    {
        Value::from_shared_object(vec![(
            Arc::from(variant),
            value.serialize(ValueSerializer)?,
        )])
    }

    fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        Ok(SequenceSerializer::new(len))
    }

    fn serialize_tuple(self, len: usize) -> Result<Self::SerializeTuple, Self::Error> {
        Ok(SequenceSerializer::new(Some(len)))
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        Ok(SequenceSerializer::new(Some(len)))
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        Ok(TupleVariantSerializer {
            variant,
            values: Vec::with_capacity(len),
        })
    }

    fn serialize_map(self, len: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
        Ok(ObjectSerializer::new(len))
    }

    fn serialize_struct(
        self,
        _name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStruct, Self::Error> {
        Ok(ObjectSerializer::new(Some(len)))
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        Ok(StructVariantSerializer {
            variant,
            object: ObjectSerializer::new(Some(len)),
        })
    }
}

struct SequenceSerializer {
    values: Vec<Value>,
}

impl SequenceSerializer {
    fn new(len: Option<usize>) -> Self {
        Self {
            values: Vec::with_capacity(len.unwrap_or(0)),
        }
    }

    fn push<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), ValueError> {
        self.values.push(value.serialize(ValueSerializer)?);
        Ok(())
    }

    fn finish(self) -> Value {
        Value::array(self.values)
    }
}

impl SerializeSeq for SequenceSerializer {
    type Ok = Value;
    type Error = ValueError;
    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Self::Error> {
        self.push(value)
    }
    fn end(self) -> Result<Self::Ok, Self::Error> {
        Ok(self.finish())
    }
}

impl SerializeTuple for SequenceSerializer {
    type Ok = Value;
    type Error = ValueError;
    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Self::Error> {
        self.push(value)
    }
    fn end(self) -> Result<Self::Ok, Self::Error> {
        Ok(self.finish())
    }
}

impl SerializeTupleStruct for SequenceSerializer {
    type Ok = Value;
    type Error = ValueError;
    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Self::Error> {
        self.push(value)
    }
    fn end(self) -> Result<Self::Ok, Self::Error> {
        Ok(self.finish())
    }
}

struct TupleVariantSerializer {
    variant: &'static str,
    values: Vec<Value>,
}

impl SerializeTupleVariant for TupleVariantSerializer {
    type Ok = Value;
    type Error = ValueError;
    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Self::Error> {
        self.values.push(value.serialize(ValueSerializer)?);
        Ok(())
    }
    fn end(self) -> Result<Self::Ok, Self::Error> {
        Value::from_shared_object(vec![(Arc::from(self.variant), Value::array(self.values))])
    }
}

struct ObjectSerializer {
    entries: Vec<(Arc<str>, Value)>,
    key: Option<Arc<str>>,
}

impl ObjectSerializer {
    fn new(len: Option<usize>) -> Self {
        Self {
            entries: Vec::with_capacity(len.unwrap_or(0)),
            key: None,
        }
    }

    fn field<T: ?Sized + Serialize>(&mut self, key: &str, value: &T) -> Result<(), ValueError> {
        self.entries
            .push((Arc::from(key), value.serialize(ValueSerializer)?));
        Ok(())
    }

    fn finish(self) -> Result<Value, ValueError> {
        Value::from_shared_object(self.entries)
    }
}

impl SerializeMap for ObjectSerializer {
    type Ok = Value;
    type Error = ValueError;

    fn serialize_key<T: ?Sized + Serialize>(&mut self, key: &T) -> Result<(), Self::Error> {
        if self.key.is_some() {
            return Err(ValueError::Unsupported("map value is missing".into()));
        }
        self.key = Some(key.serialize(KeySerializer)?);
        Ok(())
    }

    fn serialize_value<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Self::Error> {
        let key = self
            .key
            .take()
            .ok_or_else(|| ValueError::Unsupported("map key is missing".into()))?;
        self.entries.push((key, value.serialize(ValueSerializer)?));
        Ok(())
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        if self.key.is_some() {
            return Err(ValueError::Unsupported("map value is missing".into()));
        }
        self.finish()
    }
}

impl SerializeStruct for ObjectSerializer {
    type Ok = Value;
    type Error = ValueError;
    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), Self::Error> {
        self.field(key, value)
    }
    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.finish()
    }
}

struct StructVariantSerializer {
    variant: &'static str,
    object: ObjectSerializer,
}

impl SerializeStructVariant for StructVariantSerializer {
    type Ok = Value;
    type Error = ValueError;
    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), Self::Error> {
        self.object.field(key, value)
    }
    fn end(self) -> Result<Self::Ok, Self::Error> {
        Value::from_shared_object(vec![(Arc::from(self.variant), self.object.finish()?)])
    }
}

struct KeySerializer;

impl Serializer for KeySerializer {
    type Ok = Arc<str>;
    type Error = ValueError;
    type SerializeSeq = ser::Impossible<Arc<str>, ValueError>;
    type SerializeTuple = ser::Impossible<Arc<str>, ValueError>;
    type SerializeTupleStruct = ser::Impossible<Arc<str>, ValueError>;
    type SerializeTupleVariant = ser::Impossible<Arc<str>, ValueError>;
    type SerializeMap = ser::Impossible<Arc<str>, ValueError>;
    type SerializeStruct = ser::Impossible<Arc<str>, ValueError>;
    type SerializeStructVariant = ser::Impossible<Arc<str>, ValueError>;

    fn serialize_str(self, value: &str) -> Result<Self::Ok, Self::Error> {
        Ok(Arc::from(value))
    }
    fn serialize_char(self, value: char) -> Result<Self::Ok, Self::Error> {
        Ok(Arc::from(value.to_string()))
    }
    fn serialize_bool(self, value: bool) -> Result<Self::Ok, Self::Error> {
        Ok(Arc::from(value.to_string()))
    }
    fn serialize_i8(self, value: i8) -> Result<Self::Ok, Self::Error> {
        Ok(Arc::from(value.to_string()))
    }
    fn serialize_i16(self, value: i16) -> Result<Self::Ok, Self::Error> {
        Ok(Arc::from(value.to_string()))
    }
    fn serialize_i32(self, value: i32) -> Result<Self::Ok, Self::Error> {
        Ok(Arc::from(value.to_string()))
    }
    fn serialize_i64(self, value: i64) -> Result<Self::Ok, Self::Error> {
        Ok(Arc::from(value.to_string()))
    }
    fn serialize_i128(self, value: i128) -> Result<Self::Ok, Self::Error> {
        Ok(Arc::from(value.to_string()))
    }
    fn serialize_u8(self, value: u8) -> Result<Self::Ok, Self::Error> {
        Ok(Arc::from(value.to_string()))
    }
    fn serialize_u16(self, value: u16) -> Result<Self::Ok, Self::Error> {
        Ok(Arc::from(value.to_string()))
    }
    fn serialize_u32(self, value: u32) -> Result<Self::Ok, Self::Error> {
        Ok(Arc::from(value.to_string()))
    }
    fn serialize_u64(self, value: u64) -> Result<Self::Ok, Self::Error> {
        Ok(Arc::from(value.to_string()))
    }
    fn serialize_u128(self, value: u128) -> Result<Self::Ok, Self::Error> {
        Ok(Arc::from(value.to_string()))
    }
    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
    ) -> Result<Self::Ok, Self::Error> {
        Ok(Arc::from(variant))
    }
    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Self::Error> {
        value.serialize(self)
    }

    fn serialize_f32(self, _value: f32) -> Result<Self::Ok, Self::Error> {
        Err(key_error())
    }
    fn serialize_f64(self, _value: f64) -> Result<Self::Ok, Self::Error> {
        Err(key_error())
    }
    fn serialize_bytes(self, _value: &[u8]) -> Result<Self::Ok, Self::Error> {
        Err(key_error())
    }
    fn serialize_none(self) -> Result<Self::Ok, Self::Error> {
        Err(key_error())
    }
    fn serialize_some<T: ?Sized + Serialize>(self, _value: &T) -> Result<Self::Ok, Self::Error> {
        Err(key_error())
    }
    fn serialize_unit(self) -> Result<Self::Ok, Self::Error> {
        Err(key_error())
    }
    fn serialize_unit_struct(self, _name: &'static str) -> Result<Self::Ok, Self::Error> {
        Err(key_error())
    }
    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        _index: u32,
        _variant: &'static str,
        _value: &T,
    ) -> Result<Self::Ok, Self::Error> {
        Err(key_error())
    }
    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        Err(key_error())
    }
    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple, Self::Error> {
        Err(key_error())
    }
    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        Err(key_error())
    }
    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        Err(key_error())
    }
    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
        Err(key_error())
    }
    fn serialize_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStruct, Self::Error> {
        Err(key_error())
    }
    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        Err(key_error())
    }
}

fn key_error() -> ValueError {
    ValueError::Unsupported("runtime object keys must be strings or scalar identifiers".into())
}

struct OwnedSeq {
    values: std::vec::IntoIter<Value>,
}

impl<'de> SeqAccess<'de> for OwnedSeq {
    type Error = ValueError;
    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
    where
        T: de::DeserializeSeed<'de>,
    {
        self.values
            .next()
            .map(|value| seed.deserialize(value))
            .transpose()
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.values.len())
    }
}

struct OwnedMap {
    entries: std::vec::IntoIter<(Arc<str>, Value)>,
    value: Option<Value>,
}

impl<'de> MapAccess<'de> for OwnedMap {
    type Error = ValueError;
    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>, Self::Error>
    where
        K: de::DeserializeSeed<'de>,
    {
        let Some((key, value)) = self.entries.next() else {
            return Ok(None);
        };
        self.value = Some(value);
        seed.deserialize(OwnedKey(key)).map(Some)
    }
    fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value, Self::Error>
    where
        V: de::DeserializeSeed<'de>,
    {
        let value = self
            .value
            .take()
            .ok_or_else(|| ValueError::Unsupported("object value is missing".into()))?;
        seed.deserialize(value)
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.entries.len())
    }
}

struct OwnedKey(Arc<str>);

impl<'de> Deserializer<'de> for OwnedKey {
    type Error = ValueError;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_string(self.0.to_string())
    }

    fn deserialize_identifier<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_str(self.0.as_ref())
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf option unit unit_struct newtype_struct seq tuple tuple_struct
        map struct enum ignored_any
    }
}

struct ValueEnum {
    variant: String,
    value: Option<Value>,
}

impl<'de> EnumAccess<'de> for ValueEnum {
    type Error = ValueError;
    type Variant = ValueVariant;
    fn variant_seed<V>(self, seed: V) -> Result<(V::Value, Self::Variant), Self::Error>
    where
        V: de::DeserializeSeed<'de>,
    {
        let variant = seed.deserialize(self.variant.into_deserializer())?;
        Ok((variant, ValueVariant { value: self.value }))
    }
}

struct ValueVariant {
    value: Option<Value>,
}

impl<'de> VariantAccess<'de> for ValueVariant {
    type Error = ValueError;
    fn unit_variant(self) -> Result<(), Self::Error> {
        match self.value {
            None | Some(Value(Repr::Null)) => Ok(()),
            Some(_) => Err(ValueError::Unsupported("expected a unit variant".into())),
        }
    }
    fn newtype_variant_seed<T>(self, seed: T) -> Result<T::Value, Self::Error>
    where
        T: de::DeserializeSeed<'de>,
    {
        seed.deserialize(
            self.value
                .ok_or_else(|| ValueError::Unsupported("missing variant value".into()))?,
        )
    }
    fn tuple_variant<V>(self, len: usize, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        de::Deserializer::deserialize_tuple(
            self.value
                .ok_or_else(|| ValueError::Unsupported("missing tuple variant".into()))?,
            len,
            visitor,
        )
    }
    fn struct_variant<V>(
        self,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        de::Deserializer::deserialize_struct(
            self.value
                .ok_or_else(|| ValueError::Unsupported("missing struct variant".into()))?,
            "variant",
            fields,
            visitor,
        )
    }
}

impl<'de> Deserializer<'de> for Value {
    type Error = ValueError;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        match self.0 {
            Repr::Null => visitor.visit_unit(),
            Repr::Bool(value) => visitor.visit_bool(value),
            Repr::I64(value) => visitor.visit_i64(value),
            Repr::U64(value) => visitor.visit_u64(value),
            Repr::F64(value) => visitor.visit_f64(value),
            Repr::String(value) => visitor.visit_string(value.to_string()),
            Repr::Array(values) => visitor.visit_seq(OwnedSeq {
                values: Arc::try_unwrap(values)
                    .unwrap_or_else(|shared| shared.as_ref().clone())
                    .into_iter(),
            }),
            Repr::Object(entries) => visitor.visit_map(OwnedMap {
                entries: Arc::try_unwrap(entries)
                    .unwrap_or_else(|shared| shared.as_ref().clone())
                    .into_iter(),
                value: None,
            }),
        }
    }

    fn deserialize_option<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        if self.is_null() {
            visitor.visit_none()
        } else {
            visitor.visit_some(self)
        }
    }

    fn deserialize_enum<V>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        let value = match self.0 {
            Repr::String(variant) => ValueEnum {
                variant: variant.to_string(),
                value: None,
            },
            Repr::Object(entries) if entries.len() == 1 => {
                let (variant, value) = &entries[0];
                ValueEnum {
                    variant: variant.to_string(),
                    value: Some(value.clone()),
                }
            }
            _ => {
                return Err(ValueError::Unsupported(
                    "expected a string or single-key object enum".into(),
                ));
            }
        };
        visitor.visit_enum(value)
    }

    fn deserialize_newtype_struct<V>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_newtype_struct(self)
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf unit unit_struct seq tuple tuple_struct map struct identifier ignored_any
    }
}
