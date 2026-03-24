from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0003_analysishistory"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="analysishistory",
            name="recommended_cafe_type",
        ),
    ]
