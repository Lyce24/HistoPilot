import { useQueryClient } from '@tanstack/react-query';
import { api } from '../api/client';
import { workspaceKey } from '../api/queries';
import ServerFolderPicker from './ServerFolderPicker';

/** Legacy demo source registration; the shared picker only selects a server path. */
export default function FolderPicker() {
  const client = useQueryClient();
  return (
    <ServerFolderPicker
      title="Choose a WSI directory"
      label="Choose server folder"
      onSelect={async (path) => {
        await api.addSource(path);
        await client.invalidateQueries({ queryKey: workspaceKey });
      }}
    />
  );
}
