from rest_framework import serializers

class AccelerometerSerializer(serializers.Serializer):
    print('serializer check')
    values = serializers.ListField(
        child = serializers.CharField(max_length=50)
    )
    