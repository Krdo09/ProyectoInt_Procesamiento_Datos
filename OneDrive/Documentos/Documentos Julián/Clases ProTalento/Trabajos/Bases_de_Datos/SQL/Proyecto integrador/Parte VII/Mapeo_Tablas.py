from tortoise import Tortoise, run_async


var = 'python Mapeo_Tablas.py'
async def main():
    await Tortoise.init(
        db_url='postgres://postgres:Krdona 09@localhost:5432/Proyecto_int_postgresSQL',
        modules={'models': ['Modelos']},
    )
    await Tortoise.generate_schemas(safe=True)

run_async(main())
