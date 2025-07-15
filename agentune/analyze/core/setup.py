import threading

import duckdb

__setting_up = threading.Lock()
__initialized = threading.Event()

def setup() -> None: 
    """Process-wide setup that must happen before some of our code can work; the relevant code should call this idempotent method.
    
    This should NOT break any user code in the same process, even if it happens to e.g. also use duckdb.
    """
    if not __initialized.is_set():
        with __setting_up:
            if not __initialized.is_set():

                # TODO we can't INSTALL a module in production / in library mode (it downloads and installs binary libs to the user's homedir!)
                # We'll need to include the extensions in our package and tell duckdb to load them from there.
                # See: https://duckdb.org/docs/stable/extensions/installing_extensions.html
                # https://duckdb.org/docs/stable/extensions/advanced_installation_methods.html#installing-an-extension-from-an-explicit-path
                # 
                # For now we do install the extensions (and not just load them) to make development easier.

                duckdb.install_extension('spatial')

                # Note that loading an extension is in database scope (not connection scope, luckily).
                # This loads it in the global/default/in-memory database, which lets code call e.g. duckdb.dtype("geometry").
                # Every time we connect to another database, we'll also need to load it there; 
                # this is one of several reasons why we'll need a code component to manage and 'prepare' databases and connections.
                duckdb.load_extension('spatial')

                __initialized.set()

