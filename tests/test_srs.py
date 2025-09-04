import ezkl
import asyncio

async def main():
    try:
        await ezkl.get_srs('/tmp/srs.bin', logrows=17)
        print('SRS downloaded successfully')
    except Exception as e:
        print(f'Error downloading SRS: {e}')

if __name__ == '__main__':
    asyncio.run(main())

