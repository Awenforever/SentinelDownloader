# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date     : 2025/9/12 10:44 
@Author   : Jinhong Wu
@Contact  : vive@mail.ustc.edu.cn
@Project  : SpectralGPT
@File     : copernicus_downloader_v2.0.py
@IDE      : PyCharm

Copyright (c) 2025 Jinhong Wu. All rights reserved.
"""
# local
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import tomlkit
from tomlkit.exceptions import ParseError
import random
import shutil
import sys
import threading
import time
import functools
from concurrent.futures import ThreadPoolExecutor, as_completed, CancelledError
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from typing import Callable, Optional, Any
import requests
import inspect
from requests.exceptions import HTTPError, RequestException

# keyboard
import keyboard

# rich
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    DownloadColumn,
    SpinnerColumn,
    TransferSpeedColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
    TaskID,
    Task,
    filesize, ProgressSample
)
from rich.style import Style

# geopandas
import geopandas as gpd
from rich.text import Text

# shapely
from shapely.geometry import Polygon

# python-dotenv
from dotenv import load_dotenv
load_dotenv()  # doif: cannot find gdal data, modify `.env`

console = Console()

def sisyphus(
        wait: bool = False,
        min_t: int = 20,
        max_t: int = 60
):  # decorator generator/factory
    """
    The gods forced him to roll an immense boulder up a hill only for it to roll back down every time it neared the top,
    repeating this action for eternity. Through the classical influence on contemporary culture, tasks that are both
    laborious and futile are therefore described as Sisyphean.
    """
    def error_handler(func: Callable):  # decorator
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            stack = inspect.stack()
            caller_name = stack[1].function
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    wait_time = random.randint(min_t, max_t) if wait else 0
                    console.log(
                        f'[italic]Func[/] [italic bright_cyan]{func.__name__}[/] ∈ '
                        f'[italic medium_orchid]{caller_name}[/] '
                        f'@{e} || Retrying in {wait_time} s ... [red]✘'
                    )
                    time.sleep(wait_time)
        return wrapper
    return error_handler


def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance


class ThreadManager:
    def __init__(self, func: Callable, queue: list, num_workers: int):
        self._stop = threading.Event()
        self._hotkey = 'ctrl + alt + shift + p'  # pause / stop
        self._futures = {}
        self._func = func
        self._queue = queue
        self._num_workers = num_workers

    def _on_hotkey(self):
        console.log(
            f'Keyboard [italic pale_turquoise1]{self._hotkey}[/] detected: '
            f'The main thread will terminate after all child threads have finished current task.'
        )

    def _cancel(self):  # async
        self._on_hotkey()
        for f in self._futures:
            if not f.done():
                f.cancel()
        self._stop.set()

    @contextmanager
    def listening(self):
        keyboard.add_hotkey(self._hotkey, self._cancel)
        console.log(
            f'Press [italic pale_turquoise1]{self._hotkey}[/] to terminate multithreaded download tasks safely.'
        )

        with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            self._futures = {
                executor.submit(self._func, **kwargs): kwargs
                for kwargs in self._queue
            }
            try:
                yield self._futures  # __enter__ before yield; __exit__ after yield
            finally:
                keyboard.remove_hotkey(self._hotkey)
                if self._stop.is_set():
                    executor.shutdown(wait=True)
                    console.log('[italic deep_pink1]Program terminate manually.')


class WarnIf:
    def __init__(self, condition: bool):
        self.condition = condition

    def __call__(self, warning_info: str):
        if self.condition:
            stack = inspect.stack()
            caller_frame = stack[2]
            console.log(f'[bright_red]Warning[/] @[italic bright_cyan]{caller_frame.function}[/]: '
                        f'[orchid]{warning_info} [/]')


class SmartProgress(Progress):
    def update(
        self,
        task_id: TaskID,
        *,
        total: Optional[float] = None,
        completed: Optional[float] = None,
        advance: Optional[float] = None,
        description: Optional[str] = None,
        visible: Optional[bool] = None,
        refresh: bool = False,
        **fields: Any,
    ) -> None:
        with self._lock:
            task = self._tasks[task_id]
            completed_start = task.completed

            if total is not None and total != task.total:
                task.total = total
                if task.total >= SmartDownloadColumn.threshold:
                    task._reset()
            if advance is not None:
                task.completed += advance
            if completed is not None:
                task.completed = completed
            if description is not None:
                task.description = description
            if visible is not None:
                task.visible = visible
            task.fields.update(fields)
            update_completed = task.completed - completed_start

            current_time = self.get_time()
            old_sample_time = current_time - self.speed_estimate_period
            _progress = task._progress

            popleft = _progress.popleft
            while _progress and _progress[0].timestamp < old_sample_time:
                popleft()
            if update_completed > 0:
                _progress.append(ProgressSample(current_time, update_completed))
            if (
                task.total is not None
                and task.completed >= task.total
                and task.finished_time is None
            ):
                task.finished_time = task.elapsed

        if refresh:
            self.refresh()

    def reset(
        self,
        task_id: TaskID,
        *,
        start: bool = True,
        total: Optional[float] = None,
        completed: int = 0,
        visible: Optional[bool] = None,
        description: Optional[str] = None,
        **fields: Any,
    ) -> None:
        current_time = self.get_time()
        with self._lock:
            task = self._tasks[task_id]
            task._reset()
            task.start_time = current_time if start else None

            task.total = total
            task.completed = completed
            if visible is not None:
                task.visible = visible
            if fields:
                task.fields = fields
            if description is not None:
                task.description = description
            task.finished_time = None
        self.refresh()


class SmartDownloadColumn(DownloadColumn):
    threshold: int = 1024 * 1024
    def render(self, task: "Task") -> Text:
        try:
            return self._render_as_item(task)
        except (TypeError, AssertionError):
            return super().render(task)

    def _render_as_item(self, task: "Task") -> Text:
        total = int(task.total)  # int(None) -> TypeError
        assert total < self.threshold  # total >= self.threshold -> AssertionError
        completed = int(task.completed)
        item_status = f'{completed}/{total} items'
        return Text.from_markup(item_status, style="progress.download")


def _itemize_sexagesimal(it_speed: float, base: int = 60, precision: int = 1, separator: str = ' '):
    # it_speed = 0.8 it/s -> 1/0.8 s/it -> 1.25 s/it
    it_speed = 1 / (it_speed + 1e-9)
    units = ['s', 'm', 'h', 'd']  # ... 'mo' 'y'
    conversions = [1, 60, 60 * 2 * 24]
    unit, conversion = None, None

    for unit, conversion in zip(units, conversions):
        if it_speed < base * conversion:
            break

    return f"{it_speed / conversion:.{precision}f}{separator}{unit}/it"


class SmartTransferSpeedColumn(TransferSpeedColumn):
    def render(self, task: "Task") -> Text:
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text(f'?', style="progress.data.speed")

        if task.total >= SmartDownloadColumn.threshold:
            data_speed: str = filesize.decimal(int(speed)) + '/s'
        else:
            data_speed: str = _itemize_sexagesimal(speed)

        return Text(f'{data_speed}', style='progress.data.speed')


class WorkerManager:
    def __init__(self):
        self._local = threading.local()
        self._lock = threading.Lock()
        self._next_id = 0

    def get_worker_id(self) -> TaskID:
        if not hasattr(self._local, 'worker_id'):
            with self._lock:
                self._local.worker_id = TaskID(self._next_id)
                self._next_id += 1
        return self._local.worker_id


class MultiThreadCounter:
    def __init__(self, initial_value: int = 0):
        self._lock = threading.Lock()
        self._v = initial_value

    def update(self, advance: int = 1):
        with self._lock:
            self._v += advance

    @property
    def value(self) -> int:
        with self._lock:
            return self._v


class TokenManager:
    a_expire = 600 - 60
    r_expire = 3600 - 60
    def __init__(self, username: str, password: str, proxies: dict):
        self._username = username
        self._password = password
        self._proxies = proxies
        self._access_t: str = ''
        self._refresh_t: str = ''
        self._expired_t: str = ''
        self._access_time = None
        self._refresh_time = None
        self._lock = threading.RLock()  # todo: check if there is lock in lock || fixed ✔

    def _get_token(self, token_type: str) -> tuple[str, str]:
        if token_type == 'access_token':
            data = {
                'client_id': 'cdse-public',
                'username': self._username,
                'password': self._password,
                'grant_type': 'password',
            }
        elif token_type == 'refresh_token':
            data = {
                'client_id': 'cdse-public',
                'refresh_token': self._refresh_t,
                'grant_type': 'refresh_token',
            }
        else:
            raise ValueError(
                f'`token_type` must be either `access_token` or `refresh_token`, got `{token_type}` instead.'
            )
        r = requests.post(
            url="https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
            data=data,
            proxies=self._proxies,
            timeout=(10, 30)
        )
        r.raise_for_status()
        response = r.json()
        access_token, refresh_token = response.get('access_token'), response.get('refresh_token')
        if not all([access_token, refresh_token]):
            raise ValueError("No access/refresh_token in response, check the url.")
        return access_token, refresh_token

    def initialize(self):
        self._access_time, self._refresh_time = time.time(), time.time()
        self._access_t, self._refresh_t = self._get_token('access_token')
        console.log('Token initialized successfully! [green]✔')

    def _refresh_access_token(self):
        console.log('Retrieving token ...')
        self._access_time = time.time()
        self._access_t, self._refresh_t = self._get_token('refresh_token')
        console.log('Token refreshed successfully! [green]✔')

    def _check_refresh(self):
        if time.time() - self._refresh_time >= self.r_expire:
            self.initialize()

    def _check_access(self):
        if time.time() - self._access_time >= self.a_expire:
            self._refresh_access_token()

    @sisyphus(wait=True)
    def _check(self):
        self._check_refresh()
        self._check_access()

    @property
    def access_token(self) -> str:
        with self._lock:
            self._check()
            return self._access_t


class Config:
    def __set_name__(self, owner, name):
        self.name = f"_{name}"

    def __set__(self, instance, value):
        assert isinstance(value, dict)

        required_fields = {
            "authentication": ["username", "password"],
            "proxy": ["ip_port"],
            "runtime": ["save_folder"],
            "filter": ["roi_json_path"],
            "search_url": ["default"]
        }

        missing_keys = [
            f"{section}.{key}"
            for section, keys in required_fields.items()
            for key in keys
            if section not in value or key not in value[section]
        ]

        if missing_keys:
            raise RuntimeError("Missing Keys: " + ", ".join(missing_keys))

        setattr(instance, self.name, value)  # AttributeError: 'dict' object has no attribute '__dict__'

    def __get__(self, instance, owner):
        return getattr(instance, self.name)


@singleton
class SentinelDownloader:
    crs = 'EPSG: 4326'  # WGS84
    config = Config()

    def __init__(self, config_path: Path):
        # ----- load config toml -----
        try:
            with open(config_path, 'r') as file:
                config = self.config = tomlkit.load(file)
        except FileNotFoundError:
            raise ValueError(f"Configuration file does not exist: {config_path}")
        except ParseError:
            raise ValueError(f"Configuration file format error: {config_path}")

        # ----- instance property -----
        self.lock = threading.Lock()
        self.worker = WorkerManager()
        self.count = MultiThreadCounter()
        self.omit = MultiThreadCounter()
        self.progress: None | Progress = None
        self._local = threading.local()
        self._local.server_md5 = None

        # ----- configuration -----
        # identity authentication
        auth_config = config.get('authentication')
        username = auth_config.get('username')
        password = auth_config.get('password')

        # proxy
        ip_port = config.get('proxy').get('ip_port')
        self.proxies = {
            "http": ip_port,
            "https": ip_port,
        }

        # runtime settings
        runtime_config = config.get('runtime')

        # 1. save path
        save_folder = Path(runtime_config.get('save_folder'))
        self._finish = save_folder / 'Finish'
        self._temp = save_folder / 'Temp'

        # 2. threading num | default 1
        self.num_workers = runtime_config.get('num_workers', 1)

        # 3. random pause probability | set to 0.0 as `no pause`, default 0.0
        self.pause_prob = runtime_config.get('random_pause_prob', 0.0)

        # filter
        roi = Path(config.get('filter').get('roi_json_path'))
        self.roi_gdf = gpd.read_file(roi)

        # search url
        self.search_url = config.get('search_url').get('default').strip()

        # ----- initialize token -----
        self.token = TokenManager(username, password, self.proxies)
        self.token.initialize()

    @sisyphus(wait=True)
    def _rget(self, url) -> dict:
        """requests get"""
        with requests.get(url=url, proxies=self.proxies, timeout=(10, 30)) as response:
            response.raise_for_status()
            return response.json()

    def _get_total_count(self) -> int:
        return self._rget(self.search_url).get('@odata.count', 0)

    def _build_url(self, batch_size: int, skip: int):
        parsed_url = urlparse(self.search_url)
        query_params = parse_qs(parsed_url.query)
        if '$top' not in query_params or '$skip' not in query_params:
            raise ValueError('`search_url` format error, missing `$top` or `$skip` parameter.')
        query_params['$top'] = [str(batch_size)]
        query_params['$skip'] = [str(skip)]
        new_query = urlencode(query_params, doseq=True)
        return urlunparse(parsed_url._replace(query=new_query))

    def _fetch_in_batch(self, url) -> list:
        return self._rget(url).get('value', [])

    @sisyphus(wait=False)
    def search(self, batch_size: int = 900) -> list:
        search_result = []
        total = self._get_total_count()
        if total == 0:
            console.log('No data found.')
            sys.exit(0)
        console.log(f'Found {total} records. Collecting ...')
        for skip in range(0, total, batch_size):
            url = self._build_url(batch_size, skip)
            items = self._fetch_in_batch(url)
            search_result.extend([{
                'product_id': item['Id'],
                'save_path': (self._finish / item['Name']).with_suffix('.SAFE.zip'),
                'temp_path': (self._temp / item['Name']).with_suffix('.SAFE.zip'),
                'server_md5': next((check['Value'] for check in item['Checksum'] if check['Algorithm'] == 'MD5'), None),
                'geo_footprint': Polygon(item['GeoFootprint']['coordinates'][0]),
                'content_length': int(item['ContentLength'])
            } for item in items])  # todo: does items contain the MD5 and other information i need || fixed ✔
        console.log(f'Meta information for {len(search_result)} data entries has been collected. [green]✔')
        return search_result

    def _is_within_roi(self, geo_footprint: Polygon) -> bool:
        product_gdf = gpd.GeoDataFrame(geometry=[geo_footprint], crs=self.crs)
        return len(gpd.overlay(product_gdf, self.roi_gdf, how='intersection')) > 0

    def _hash_check(self, file_path: Path) -> bool:
        h = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest() == self._local.server_md5

    def _file_check(self, save_path: Path, temp_path: Path) -> bool:
        if temp_path.exists():
            if self._hash_check(temp_path):
                shutil.move(temp_path, save_path)
                console.log(f'Download completed: [italic indian_red]{save_path.name}[/]. [green]✔')
                self.count.update()
                return True
            else:
                temp_path.unlink()
        return False

    def _wait(self, task_id: TaskID):
        wait_time = random.randint(20, 60)
        console.log(f'Retrying in {wait_time} s ...')
        self.progress.reset(
            task_id=task_id,
            description=f'Thread-{task_id} || On hold...',
            total=wait_time
        )

        for _ in range(wait_time):
            self.progress.update(task_id=task_id, advance=1)
            time.sleep(1)
        self.progress.reset(task_id=task_id)

    @staticmethod
    def _prometheus(func: Callable):
        @functools.wraps(func)
        def error_handler(self: SentinelDownloader, **kwargs):
            task_id = self.worker.get_worker_id()
            completed = False
            while not completed:  # uncaught exception will break this while loop anyway
                try:
                    func(self, **kwargs)
                except HTTPError as e:
                    console.log(f'Thread-{task_id} || [italic]Failed[/] @{e}. [red]✘')
                except RequestException as e:
                    console.log(f'Thread-{task_id} || [italic]Failed[/] @{e}. [red]✘')
                    self._wait(task_id)
                finally:
                    self.progress.reset(
                        task_id=task_id,
                        description=f'Thread-{task_id} || Pending...',
                    )
                    completed = self._file_check(
                        kwargs.get('_save_path'), kwargs.get('temp_path')
                    )

        return error_handler

    @_prometheus
    def _write(self, product_id: str, _save_path: Path, temp_path: Path, file_size: int):
        download_url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
        session = requests.Session()
        session.headers.update({'Authorization': f'Bearer {self.token.access_token}'})
        task_id = self.worker.get_worker_id()

        with session.get(url=download_url, stream=True, proxies=self.proxies, timeout=(30, 60)) as response:
            response.raise_for_status()
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        self.progress.update(
                            task_id=task_id,
                            description=temp_path.name,
                            advance=len(chunk),
                            total=file_size,
                            visible=True
                        )

    def _random_pause(self, prob: float = 0.5, p_min: int = 60, p_max: int = 600, seed: int | float = time.time()):
        WarnIf(p_max >= SmartDownloadColumn.threshold)(
            f'For correct progress bar display, `p_max` should be less than {SmartDownloadColumn.threshold} '
            f'(currently: {p_max})"'
        )
        task_id = self.worker.get_worker_id()
        rng = random.Random(seed + task_id)
        p = rng.random()
        wait_time = rng.randint(p_min, p_max)
        if p < prob:
            self.progress.reset(
                task_id=task_id,
                description=f'Thread-{task_id} || Pausing ...',
                total=wait_time
            )
            for _ in range(wait_time):
                self.progress.update(task_id=task_id, advance=1)
                time.sleep(1)
            self.progress.reset(task_id=task_id)

    def _download_single_file(
            self,
            product_id: str,
            save_path: Path,
            temp_path: Path,
            server_md5: str,
            geo_footprint: Polygon,
            content_length: int
    ) -> None:
        if save_path.exists():
            self.count.update()
            return
        if not self._is_within_roi(geo_footprint):
            self.omit.update()
            return

        time.sleep(random.uniform(0, 3))
        self._local.server_md5 = server_md5
        self._write(
            product_id=product_id,
            _save_path=save_path,
            temp_path=temp_path,
            file_size=content_length,
        )

        self._random_pause(prob=self.pause_prob)

    def _prepare_directories(self):
        self._finish.mkdir(parents=True, exist_ok=True)
        self._temp.mkdir(parents=True, exist_ok=True)

    def multi_download(self):
        self._prepare_directories()

        with SmartProgress(
                SpinnerColumn('dots'),
                TextColumn("[bold blue]{task.description}", justify="right"),
                SmartDownloadColumn(),
                BarColumn(
                    bar_width=None,
                    complete_style=Style(color="green"),
                    finished_style=Style(color="cyan"),
                    pulse_style=Style(color="light_goldenrod1")
                ),
                TaskProgressColumn(
                    "[progress.percentage]{task.percentage:>3.1f}%"
                ),
                "• 🚀",
                SmartTransferSpeedColumn(),
                "• ⌚ [yellow]",
                TimeElapsedColumn(),
                "• ⏳ [cyan]",
                TimeRemainingColumn(),
                console=console,
                expand=True,
                speed_estimate_period=60.0 * 60,
        ) as self.progress:
            [
                self.progress.add_task(description=f'Thread-{i}...', visible=False)
                for i in range(self.num_workers + 1)
            ]

            download_queue: list = self.search()
            total = len(download_queue)
            with ThreadManager(
                    self._download_single_file,
                    download_queue,
                    self.num_workers
            ).listening() as executions:
                for future in as_completed(executions):
                    try:
                        future.result()
                    except CancelledError:
                        pass
                    except Exception as e:
                        console.log(f"Unhandled error in future: {e}")

                    completed, omitted = self.count.value, self.omit.value
                    self.progress.update(
                        task_id=self.num_workers,
                        description=f'Downloading {total} @',
                        completed=completed,
                        total=total - omitted,
                        visible=True,
                    )

        console.log('All done.')


if __name__ == '__main__':
    config_toml = Path('./cfg.toml')  # duplicate `cfg_example.toml` and customize
    downloader = SentinelDownloader(config_toml)
    downloader.multi_download()
