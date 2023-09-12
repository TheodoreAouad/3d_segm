[webdav] Fatal: OCP\Files\NotFoundException: /admin_hexanet/files/Photos/ARODE_MALIN_01 at <<closure>>

 0. /var/www/datachallenge2022sfr.fr/htdocs/nextcloud/lib/private/Files/Node/LazyUserFolder.php line 59
    OC\Files\Node\Root->get()
 1. /var/www/datachallenge2022sfr.fr/htdocs/nextcloud/lib/private/Encryption/File.php line 94
    OC\Files\Node\LazyUserFolder->get()
 2. /var/www/datachallenge2022sfr.fr/htdocs/nextcloud/lib/private/Files/Stream/Encryption.php line 284
    OC\Encryption\File->getAccessList()
 3. <<closure>>
    OC\Files\Stream\Encryption->stream_open()
 4. /var/www/datachallenge2022sfr.fr/htdocs/nextcloud/lib/private/Files/Stream/Encryption.php line 213
    fopen()
 5. /var/www/datachallenge2022sfr.fr/htdocs/nextcloud/lib/private/Files/Stream/Encryption.php line 188
    OC\Files\Stream\Encryption::wrapSource()
 6. /var/www/datachallenge2022sfr.fr/htdocs/nextcloud/lib/private/Files/Storage/Wrapper/Encryption.php line 470
    OC\Files\Stream\Encryption::wrap()
 7. /var/www/datachallenge2022sfr.fr/htdocs/nextcloud/lib/private/Files/Storage/Wrapper/Encryption.php line 818
    OC\Files\Storage\Wrapper\Encryption->fopen()
 8. /var/www/datachallenge2022sfr.fr/htdocs/nextcloud/lib/private/Files/Storage/Wrapper/Encryption.php line 690
    OC\Files\Storage\Wrapper\Encryption->copyBetweenStorage()
 9. /var/www/datachallenge2022sfr.fr/htdocs/nextcloud/lib/private/Files/Storage/Wrapper/Encryption.php line 811
    OC\Files\Storage\Wrapper\Encryption->copyFromStorage()
10. /var/www/datachallenge2022sfr.fr/htdocs/nextcloud/lib/private/Files/Storage/Wrapper/Encryption.php line 690
    OC\Files\Storage\Wrapper\Encryption->copyBetweenStorage()
11. /var/www/datachallenge2022sfr.fr/htdocs/nextcloud/lib/private/Files/Storage/Wrapper/Wrapper.php line 581
    OC\Files\Storage\Wrapper\Encryption->copyFromStorage()
12. /var/www/datachallenge2022sfr.fr/htdocs/nextcloud/lib/private/Files/View.php line 945
    OC\Files\Storage\Wrapper\Wrapper->copyFromStorage()
13. /var/www/datachallenge2022sfr.fr/htdocs/nextcloud/apps/dav/lib/Connector/Sabre/Directory.php line 486
    OC\Files\View->copy()
14. /var/www/datachallenge2022sfr.fr/htdocs/nextcloud/3rdparty/sabre/dav/lib/DAV/Tree.php line 132
    OCA\DAV\Connector\Sabre\Directory->copyInto()
15. /var/www/datachallenge2022sfr.fr/htdocs/nextcloud/3rdparty/sabre/dav/lib/DAV/CorePlugin.php line 659
    Sabre\DAV\Tree->copy()
16. /var/www/datachallenge2022sfr.fr/htdocs/nextcloud/3rdparty/sabre/event/lib/WildcardEmitterTrait.php line 89
    Sabre\DAV\CorePlugin->httpCopy()
17. /var/www/datachallenge2022sfr.fr/htdocs/nextcloud/3rdparty/sabre/dav/lib/DAV/Server.php line 472
    Sabre\DAV\Server->emit()
18. /var/www/datachallenge2022sfr.fr/htdocs/nextcloud/3rdparty/sabre/dav/lib/DAV/Server.php line 253
    Sabre\DAV\Server->invokeMethod()
19. /var/www/datachallenge2022sfr.fr/htdocs/nextcloud/3rdparty/sabre/dav/lib/DAV/Server.php line 321
    Sabre\DAV\Server->start()
20. /var/www/datachallenge2022sfr.fr/htdocs/nextcloud/apps/dav/lib/Server.php line 358
    Sabre\DAV\Server->exec()
21. /var/www/datachallenge2022sfr.fr/htdocs/nextcloud/apps/dav/appinfo/v2/remote.php line 35
    OCA\DAV\Server->exec()
22. /var/www/datachallenge2022sfr.fr/htdocs/nextcloud/remote.php line 170
    require_once("/var/www/datach ... p")

COPY /remote.php/dav/files/admin_hexanet/ARODE_MALIN_01
from 81.23.32.9 by admin_hexanet at 2023-08-16T13:16:50+00:00